"""Chapter 02 · Derivatives & gradient descent on the cars.

The baseline driver has a knob `aggression`: how fast it dares to go for a given free distance ahead.
Loss J(a) = average lap time over the 10 training tracks (a crash counts as 30 s).
- too timid  → slow laps
- too brave  → crashes
The derivative dJ/da is the slope of that curve. We estimate it numerically (central difference, step 02)
and walk downhill:  a ← a − lr · dJ/da.

Run:  uv run python -m chapters.ch02_gradient            (plots + GIF + results_ch02.json)
      uv run python -m chapters.ch02_gradient --watch    (live window: cars from different GD steps race)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from racing.car import Cars
from racing.drivers import Baseline
from racing.render import COLORS, MUTED, Canvas
from racing.sim import lap
from racing.track import TRAIN_SEEDS, generate

CRASH_TIME = 30.0
TRACKS = [generate(s) for s in TRAIN_SEEDS]


def J(a: float) -> float:
    times = []
    for t in TRACKS:
        r = lap(Baseline(aggression=a), t, max_time=CRASH_TIME)
        times.append(r.lap_time if r.finished else CRASH_TIME)
    return float(np.mean(times))


def gradient_descent(a0=0.4, lr0=0.01, decay=0.75, h=0.05, steps=10) -> list[dict]:
    a, path = a0, []
    for k in range(steps):
        j = J(a)
        grad = (J(a + h) - J(a - h)) / (2 * h)  # central difference
        lr = lr0 * decay**k  # shrinking steps: near the crash cliff the gradient jumps around
        path.append({"step": k, "a": a, "J": j, "grad": grad, "lr": lr})
        print(f"step {k:2d}  a = {a:.3f}  J = {j:6.2f} s  dJ/da = {grad:+7.2f}")
        a = float(np.clip(a - lr * grad, 0.2, 2.5))
    path.append({"step": steps, "a": a, "J": J(a), "grad": float("nan")})
    print(f"final    a = {a:.3f}  J = {path[-1]['J']:6.2f} s")
    return path


def plot(curve, path) -> None:
    a = np.array([c["a"] for c in curve])
    j = np.array([c["J"] for c in curve])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(a, j, color="#64748b", lw=2, label="J(a): mean lap time on 10 training tracks")
    for p in path[:-1:3]:  # tangent at a few GD steps: slope = derivative
        xs = np.array([p["a"] - 0.12, p["a"] + 0.12])
        ax.plot(xs, p["J"] + p["grad"] * (xs - p["a"]), color="#dc2626", lw=1.5)
    ax.plot([p["a"] for p in path], [p["J"] for p in path], "o-", color="#2563eb", label="gradient descent steps")
    for p in (path[0], path[-1]):
        ax.annotate(f"step {p['step']}: a={p['a']:.2f}, {p['J']:.1f} s", (p["a"], p["J"]), textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.axvspan(1.35, a.max(), color="#fee2e2", alpha=0.6, label="cars crash on most tracks")
    ax.set(xlabel="aggression a", ylabel="J(a) = mean lap time [s]", title="Red lines: tangents (slope = dJ/da). Gradient descent walks downhill.")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig("assets/ch02_gradient.png", dpi=150)


def race(path, watch: bool) -> None:
    """Cars with the aggression from GD steps 0, middle and final race on one track side by side."""
    picks = [path[0], path[len(path) // 2], path[-1]]
    track = TRACKS[2]
    aggr = np.array([p["a"] for p in picks])
    driver = Baseline(aggression=aggr)
    cars = Cars(track, len(picks))
    cv = Canvas(watch, "Ch. 02 · Gradient descent")
    trails = [[] for _ in picks]
    for _ in range(int(20 * 30)):
        obs = cars.observe()
        cars.step(*driver(obs))
        cv.track(track)
        lines = [f"J(a) = mean lap time, 10 tracks", f"a ← a − lr · dJ/da, lr: 0.01 → shrinking", ""]
        for k, (p, col) in enumerate(zip(picks, COLORS)):
            trails[k].append(cars.pos[k].copy())
            cv.path(trails[k][-90:], col, 2)
            cv.car(cars.pos[k], cars.heading[k], col, scale=1.3)
            state = f"{cars.lap_time[k]:.2f} s ✓" if cars.finished[k] else ("crashed" if not cars.alive[k] else f"{cars.time[k]:.1f} s")
            lines.append((f"GD step {p['step']:>2}: a = {p['a']:.2f}", col))
            lines.append((f"   J = {p['J']:.2f} s   this lap: {state}", col))
        lines += ["", ("blue = start, red = middle,", MUTED), ("green = after gradient descent", MUTED)]
        cv.panel("02 · Gradient descent", lines)
        if not cv.show() or (cars.finished | ~cars.alive).all():
            break
    for _ in range(20):  # hold the last frame
        cv.show()
    if not watch:
        cv.save_gif("assets/ch02_gradient.gif")


def main(watch: bool) -> None:
    curve = [{"a": float(a), "J": J(a)} for a in np.arange(0.3, 2.01, 0.05)]
    path = gradient_descent()
    json.dump({"curve": curve, "path": path}, open("results_ch02.json", "w"), indent=2)
    plot(curve, path)
    race(path, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
