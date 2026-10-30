"""Car physics and ray sensors, vectorised over many cars at once.

State per car: position (x, y), heading angle, speed.
Controls per car: steer in [-1, 1] (positive = left), throttle in [-1, 1] (negative = brake).
"""

import numpy as np

from racing.track import Track

RAY_ANGLES = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])  # relative to heading; positive = left
RAY_RANGE = 220.0
MAX_SPEED = 420.0  # px / s
ACCEL = 300.0
DRAG = 0.6
TURN_RATE = 2.6  # rad / s at full steer
DT = 1 / 30


def rotation(theta: np.ndarray) -> np.ndarray:
    """(N,) angles → (N, 2, 2) rotation matrices [[cos, -sin], [sin, cos]]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def ray_directions(heading: np.ndarray) -> np.ndarray:
    """(N,) headings → (N, R, 2) unit ray directions = R(heading + offset) @ [1, 0]."""
    angles = heading[:, None] + RAY_ANGLES[None, :]  # (N, R)
    return rotation(angles.ravel()).reshape(*angles.shape, 2, 2)[..., 0]  # first column = R @ [1, 0]


def cast_rays(pos: np.ndarray, heading: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Distance from each car along each ray to the nearest wall, capped at RAY_RANGE.

    Ray p + t·d meets segment a + u·(b − a) where  t·d − u·(b − a) = a − p.
    That is a 2×2 linear system; Cramer's rule solves it for all rays × walls at once.
    det = 0 means the ray is parallel to the wall (no intersection).
    """
    d = ray_directions(heading)  # (N, R, 2)
    a, b = walls[:, 0], walls[:, 1]  # (W, 2)
    e = b - a  # (W, 2)
    ap = a[None, None] - pos[:, None, None]  # (N, 1, W, 2)
    dd = d[:, :, None]  # (N, R, 1, 2)
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
    det = cross(dd, -e[None, None])  # (N, R, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(ap, -e[None, None]) / det
        u = cross(dd, ap) / det
    hit = (np.abs(det) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, RAY_RANGE)
    return np.minimum(t.min(axis=-1), RAY_RANGE)  # (N, R)


class Cars:
    """N cars driving on one track."""

    def __init__(self, track: Track, n: int = 1):
        self.track = track
        p0, h0 = track.start
        self.pos = np.repeat(p0[None], n, 0).astype(float)
        self.heading = np.full(n, h0)
        self.speed = np.zeros(n)
        self.alive = np.ones(n, bool)
        idx, _ = track.nearest(p0)
        self.idx = np.full(n, idx)  # nearest centerline index
        self.progress = np.zeros(n)  # distance driven along the centerline (px)
        self.time = np.zeros(n)
        self.finished = np.zeros(n, bool)
        self.lap_time = np.full(n, np.nan)

    @property
    def n(self) -> int:
        return len(self.speed)

    def observe(self) -> np.ndarray:
        """Inputs for a driver: 7 ray distances and speed, all scaled to [0, 1] → (N, 8)."""
        rays = cast_rays(self.pos, self.heading, self.track.walls) / RAY_RANGE
        return np.concatenate([rays, (self.speed / MAX_SPEED)[:, None]], axis=1)

    def step(self, steer: np.ndarray, throttle: np.ndarray) -> None:
        live = self.alive & ~self.finished
        steer, throttle = np.clip(steer, -1, 1), np.clip(throttle, -1, 1)
        self.speed = np.where(live, np.clip(self.speed + (throttle * ACCEL - DRAG * self.speed) * DT, 0, MAX_SPEED), self.speed)
        self.heading = np.where(live, self.heading + steer * TURN_RATE * DT, self.heading)
        vel = np.stack([np.cos(self.heading), np.sin(self.heading)], 1) * self.speed[:, None]
        self.pos = np.where(live[:, None], self.pos + vel * DT, self.pos)
        self.time = np.where(live, self.time + DT, self.time)
        self._update_progress(live)

    def _update_progress(self, live: np.ndarray) -> None:
        c, N = self.track.center, len(self.track.center)
        for i in np.flatnonzero(live):
            # search only near the last index: cheap and can't jump across the track
            window = (self.idx[i] + np.arange(-10, 30)) % N
            d = np.linalg.norm(c[window] - self.pos[i], axis=1)
            j = int(np.argmin(d))
            new = int(window[j])
            delta = (new - self.idx[i] + N // 2) % N - N // 2  # signed index change, wraps around
            if delta > 0:
                self.progress[i] += self.track.seg_len[(self.idx[i] + np.arange(delta)) % N].sum()
            elif delta < 0:
                self.progress[i] -= self.track.seg_len[(new + np.arange(-delta)) % N].sum()
            self.idx[i] = new
            if d[j] > self.track.width / 2:  # left the track
                self.alive[i] = False
            if self.progress[i] >= self.track.length:
                self.finished[i] = True
                self.lap_time[i] = self.time[i]
