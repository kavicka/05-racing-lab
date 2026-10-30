"""Headless simulation: run drivers on a track without a window."""

from dataclasses import dataclass

import numpy as np

from racing.car import DT, Cars
from racing.track import Track


@dataclass
class LapResult:
    finished: bool
    lap_time: float  # seconds; nan if not finished
    progress: float  # fraction of the lap driven, 0..1
    trajectory: np.ndarray  # (T, 2) positions of car 0


def run(driver, track: Track, n: int = 1, max_time: float = 60.0, record: bool = False, on_step=None) -> tuple[Cars, list]:
    cars = Cars(track, n)
    traj = []
    for _ in range(int(max_time / DT)):
        if not (cars.alive & ~cars.finished).any():
            break
        obs = cars.observe()
        steer, throttle = driver(obs)
        cars.step(steer, throttle)
        if record:
            traj.append(cars.pos.copy())
        if on_step:
            on_step(cars, obs, steer, throttle)
    return cars, traj


def lap(driver, track: Track, max_time: float = 60.0) -> LapResult:
    cars, traj = run(driver, track, 1, max_time, record=True)
    return LapResult(bool(cars.finished[0]), float(cars.lap_time[0]), float(min(cars.progress[0] / track.length, 1.0)),
                     np.array([p[0] for p in traj]))
