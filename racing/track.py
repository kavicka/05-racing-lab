"""Procedurally generated closed race tracks.

A track is a smooth closed centerline plus a constant width. The walls are the
centerline shifted left/right along its normals, stored as line segments.
Coordinates are math-style (y up); only the renderer flips y.
"""

import json
from dataclasses import dataclass, field

import numpy as np

WORLD_W, WORLD_H = 900, 640


@dataclass
class Track:
    center: np.ndarray  # (N, 2) centerline points, counter-clockwise
    width: float
    seed: int = -1
    left: np.ndarray = field(init=False)  # (N, 2) left wall points
    right: np.ndarray = field(init=False)
    walls: np.ndarray = field(init=False)  # (2N, 2, 2) wall segments [start, end]
    seg_len: np.ndarray = field(init=False)  # (N,) length of each centerline segment
    length: float = field(init=False)

    def __post_init__(self):
        c = self.center
        tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # tangent rotated +90° (points left)
        self.left = c + normal * self.width / 2
        self.right = c - normal * self.width / 2
        segs = []
        for edge in (self.left, self.right):
            segs.append(np.stack([edge, np.roll(edge, -1, axis=0)], axis=1))
        self.walls = np.concatenate(segs)
        self.seg_len = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
        self.length = float(self.seg_len.sum())

    @property
    def start(self) -> tuple[np.ndarray, float]:
        d = self.center[1] - self.center[0]
        return self.center[0].copy(), float(np.arctan2(d[1], d[0]))

    def nearest(self, pos: np.ndarray) -> tuple[int, float]:
        """Index of the closest centerline point and the distance to it."""
        d = np.linalg.norm(self.center - pos, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def save(self, path: str) -> None:
        json.dump({"center": self.center.tolist(), "width": self.width, "seed": self.seed}, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "Track":
        d = json.load(open(path))
        return cls(np.array(d["center"]), d["width"], d.get("seed", -1))


def generate(seed: int, n_points: int = 360, width: float = 60.0) -> Track:
    """Radius as a sum of a few random harmonics → a smooth, non-self-intersecting loop."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = np.ones_like(theta)
    for k in (2, 3, 4, 5, 6):
        amp = rng.uniform(0.04, 0.3 / k**0.5)
        r += amp * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    rx, ry = WORLD_W * 0.36, WORLD_H * 0.34
    center = np.stack([WORLD_W / 2 + rx * r * np.cos(theta), WORLD_H / 2 + ry * r * np.sin(theta)], axis=1)
    return Track(center, width, seed)


TRAIN_SEEDS = list(range(0, 10))
TEST_SEEDS = list(range(100, 110))
