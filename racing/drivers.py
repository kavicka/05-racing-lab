"""Drivers turn observations (N, 8) into controls: steer (N,), throttle (N,)."""

import numpy as np


class Baseline:
    """Hand-written driver: steer toward the side with more room, slow down when the road ahead is short.

    steer_gain: how hard it steers toward open space (step 02 optimises this)
    aggression: how fast it dares to go for a given free distance ahead
    """

    def __init__(self, steer_gain: float = 3.0, aggression: float = 0.8):
        self.steer_gain = steer_gain
        self.aggression = aggression

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays, speed = obs[:, :7], obs[:, 7]
        left = rays[:, 4:].mean(1)  # +30°, +60°, +90°
        right = rays[:, :3].mean(1)  # −90°, −60°, −30°
        steer = self.steer_gain * (left - right)
        target = np.clip(rays[:, 3] * self.aggression * 1.4, 0.25, 1.0)  # target speed from free distance ahead
        throttle = np.clip((target - speed) * 5, -1, 1)
        return steer, throttle


class MLPBatch:
    """N tiny MLPs (8 → H → 2) evaluated either one by one or as one batched matmul (step 01)."""

    def __init__(self, n: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(8), (n, 8, hidden))
        self.b1 = np.zeros((n, hidden))
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (n, hidden, 2))
        self.b2 = np.array([0.0, 0.6]) + np.zeros((n, 2))  # bias toward driving forward

    def loop(self, obs: np.ndarray) -> np.ndarray:
        out = np.empty((len(obs), 2))
        for i in range(len(obs)):  # one car at a time
            h = np.tanh(obs[i] @ self.W1[i] + self.b1[i])
            out[i] = np.tanh(h @ self.W2[i] + self.b2[i])
        return out

    def batched(self, obs: np.ndarray) -> np.ndarray:
        # (N, 1, 8) @ (N, 8, H) → (N, 1, H): every car's network in one call
        h = np.tanh((obs[:, None] @ self.W1)[:, 0] + self.b1)
        return np.tanh((h[:, None] @ self.W2)[:, 0] + self.b2)

    def __call__(self, obs):
        out = self.batched(obs)
        return out[:, 0], out[:, 1]
