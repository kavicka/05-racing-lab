"""Chapter 03 · Vectors & matrices on the cars.

What you see:
- blue arrow: the heading vector h = R(θ) @ [1, 0]  (a rotation matrix applied to "forward")
- green arrow: the track direction t at the nearest centerline point
- the dot product h · t = cos(angle between them): 1 = perfectly aligned, 0 = driving across the track
- yellow lines: 7 rays, each R(θ + offset) @ [1, 0], cut where they hit a wall
  (ray–wall intersection = a 2×2 linear system, solved with determinants: det = 0 → parallel)

Run:  uv run python -m chapters.ch03_vectors --watch     (live window)
      uv run python -m chapters.ch03_vectors             (writes assets/ch03_vectors.gif)
"""

import argparse

import numpy as np

from racing.car import RAY_ANGLES, Cars, cast_rays, ray_directions
from racing.drivers import Baseline
from racing.render import COLORS, MUTED, Canvas
from racing.track import generate


def main(watch: bool, seconds: float = 12.0) -> None:
    track = generate(3)
    cars = Cars(track, 1)
    driver = Baseline()
    cv = Canvas(watch, "Ch. 03 · Vectors & matrices")
    trail = []
    for _ in range(int(seconds * 30)):
        obs = cars.observe()
        steer, throttle = driver(obs)
        cars.step(steer, throttle)
        if not cars.alive[0] or cars.finished[0]:
            cars = Cars(track, 1)
            trail = []
        p, th = cars.pos[0], cars.heading[0]
        trail.append(p.copy())

        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        h = R @ np.array([1.0, 0.0])
        i = cars.idx[0]
        t = track.center[(i + 1) % len(track.center)] - track.center[i - 1]
        t /= np.linalg.norm(t)
        dot = float(h @ t)
        dists = cast_rays(p[None], np.array([th]), track.walls)[0]
        dirs = ray_directions(np.array([th]))[0]

        cv.track(track)
        cv.path(trail[-120:], (148, 163, 184), 2)
        cv.rays(p, dirs, dists)
        cv.arrow(p, t * 95, COLORS[2], 4, "t")
        cv.arrow(p, h * 95, COLORS[0], 4, "h")
        cv.car(p, th, COLORS[1], scale=1.5)

        y = cv.panel("03 · Vectors & matrices", [
            f"heading θ = {np.degrees(th) % 360:6.1f}°",
            "rotation matrix R(θ):",
            f"  [ {R[0, 0]:+.2f}  {R[0, 1]:+.2f} ]",
            f"  [ {R[1, 0]:+.2f}  {R[1, 1]:+.2f} ]",
            f"h = R(θ)·[1,0] = [{h[0]:+.2f}, {h[1]:+.2f}]",
            (f"h · t = cos α = {dot:+.3f}", COLORS[2]),
            (f"angle to track α = {np.degrees(np.arccos(np.clip(dot, -1, 1))):5.1f}°", COLORS[2]),
            f"speed |v| = {cars.speed[0]:5.0f} px/s",
            ("ray hit: 2×2 linear system", MUTED),
        ])
        for k, (ang, dist) in enumerate(zip(np.degrees(RAY_ANGLES), dists)):
            cv.bar(cv.screen.get_width() - 314, y + 22 + k * 30, 290, dist, 0, 220, (250, 204, 21), f"ray {ang:+.0f}°  {dist:5.0f} px")
        if not cv.show():
            break
    if not watch:
        cv.save_gif("assets/ch03_vectors.gif")
        print("wrote assets/ch03_vectors.gif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
