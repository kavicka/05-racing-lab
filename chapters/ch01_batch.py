"""Chapter 01 · Why batching matters: one car vs hundreds of cars.

Every car has its own tiny neural network brain (8 → 16 → 2). Computing the brains
- one car at a time: a Python loop with N small matrix products
- all at once: one batched matmul (N, 1, 8) @ (N, 8, 16)
gives the same steering, but the time per simulation step is very different.
On a Mac the same batched brains also run on the GPU (MPS) via PyTorch.

Run:  uv run python -m chapters.ch01_batch            (benchmark + plot + GIF)
      uv run python -m chapters.ch01_batch --watch    (live window with 200 cars)
"""

import argparse
import json
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from racing.car import Cars
from racing.drivers import MLPBatch
from racing.render import COLORS, MUTED, Canvas
from racing.track import generate

SIZES = [1, 10, 50, 100, 200, 500, 1000, 2000]


def bench(fn, repeats: int = 20) -> float:
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats * 1000  # ms


def torch_brain(brain: MLPBatch, device: str):
    W1, b1, W2, b2 = (torch.tensor(x, dtype=torch.float32, device=device) for x in (brain.W1, brain.b1, brain.W2, brain.b2))

    def f(obs: torch.Tensor):
        h = torch.tanh(torch.bmm(obs[:, None], W1)[:, 0] + b1)
        out = torch.tanh(torch.bmm(h[:, None], W2)[:, 0] + b2)
        if device == "mps":
            torch.mps.synchronize()  # GPU work is asynchronous (step 01)
        return out

    return f


def benchmark() -> dict:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    rows = []
    for n in SIZES:
        brain = MLPBatch(n)
        obs = np.random.default_rng(0).random((n, 8))
        assert np.allclose(brain.loop(obs), brain.batched(obs))  # same answer both ways
        row = {"n": n, "loop_ms": bench(lambda: brain.loop(obs), 5 if n >= 1000 else 20), "batched_ms": bench(lambda: brain.batched(obs))}
        for d in devices:
            f, x = torch_brain(brain, d), torch.tensor(obs, dtype=torch.float32, device=d)
            row[f"torch_{d}_ms"] = bench(lambda: f(x))
        rows.append(row)
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return {"platform": platform.platform(), "devices": devices, "rows": rows}


def plot(res: dict) -> None:
    rows = res["rows"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.loglog(n, [r["loop_ms"] for r in rows], "o-", label="NumPy, Python loop over cars")
    ax.loglog(n, [r["batched_ms"] for r in rows], "s-", label="NumPy, one batched matmul")
    for d in res["devices"]:
        ax.loglog(n, [r[f"torch_{d}_ms"] for r in rows], "^-", label=f"PyTorch batched on {d.upper()}")
    ax.axhline(1000 / 30, color="#dc2626", ls="--", lw=1)
    ax.text(1.1, 1000 / 30 * 1.15, "33 ms = whole frame budget at 30 FPS", color="#dc2626", fontsize=8)
    ax.set(xlabel="number of cars", ylabel="time to compute all brains [ms] (log)", title="Brains for N cars: loop vs one batched matrix multiply")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/ch01_batch.png", dpi=150)


def swarm(res: dict, watch: bool, n: int = 200) -> None:
    track = generate(5)
    brain = MLPBatch(n, seed=7)
    cars = Cars(track, n)
    cv = Canvas(watch, "Ch. 01 · 200 brains, one matmul")
    r200 = next(r for r in res["rows"] if r["n"] == 200)
    for frame in range(int(12 * 30)):
        obs = cars.observe()
        t = time.perf_counter()
        out = brain.batched(obs)
        brain_ms = (time.perf_counter() - t) * 1000
        cars.step(out[:, 0], out[:, 1])
        cv.track(track)
        for i in np.flatnonzero(cars.alive):
            cv.car(cars.pos[i], cars.heading[i], COLORS[i % len(COLORS)], scale=0.8)
        best = int(np.argmax(cars.progress))
        cv.panel("01 · Batching", [
            f"{n} cars, each with its own brain",
            "brain: MLP 8 → 16 → 2, random",
            f"alive: {cars.alive.sum():3d} / {n}",
            f"best progress: {cars.progress[best] / track.length:5.1%} of a lap",
            "",
            ("all brains in ONE batched matmul:", COLORS[2]),
            ("  (N,1,8) @ (N,8,16) → (N,1,16)", COLORS[2]),
            f"  this frame: {brain_ms:.2f} ms",
            "",
            ("benchmark for 200 cars (this machine):", MUTED),
            f"  Python loop:   {r200['loop_ms']:6.2f} ms",
            f"  batched NumPy: {r200['batched_ms']:6.2f} ms",
            f"  → {r200['loop_ms'] / r200['batched_ms']:.0f}× faster, same result",
            "",
            ("random brains mostly crash:", MUTED),
            ("chapter 'evolution' will breed the best", MUTED),
        ])
        if not cv.show() or not cars.alive.any():
            break
    if not watch:
        cv.save_gif("assets/ch01_batch.gif")


def main(watch: bool) -> None:
    res = benchmark()
    json.dump(res, open("results_ch01.json", "w"), indent=2)
    plot(res)
    swarm(res, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
