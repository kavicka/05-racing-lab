"""Chapter 02b · The chain rule on a braking car (backprop through time).

A car arrives at v0 = 300 px/s and brakes with a constant force b. Where does it stop?
Physics, one step of dt (k = 0 … N−1):
    v[k+1] = max(0, v[k] − b·dt − DRAG·v[k]·dt)
    x[k+1] = x[k] + v[k+1]·dt
Loss: L(b) = (x[N] − target)²  →  stop exactly on the line.

x[N] depends on b through 90 nested steps, so dL/db is a long chain:
    dL/db = dL/dx[N] · Σ_k dx[N]/dv[k] · dv[k]/db
We compute it three ways, exactly like step 02 did on e = a·b, d = e + c, L = d²:
  1. by hand: a backward loop over time steps (the chain rule written out)
  2. numerically: (L(b+h) − L(b−h)) / 2h
  3. PyTorch autograd
and then use it for gradient descent on b.

Run:  uv run python -m chapters.ch02b_chain_rule            (plot + GIF + results_ch02b.json)
      uv run python -m chapters.ch02b_chain_rule --watch
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from racing.render import COLORS, MUTED, Canvas

V0, TARGET, DT, N, DRAG = 300.0, 250.0, 1 / 30, 90, 0.6


def forward(b: float):
    v, x = [V0], [0.0]
    for _ in range(N):
        v.append(max(0.0, v[-1] - b * DT - DRAG * v[-1] * DT))
        x.append(x[-1] + v[-1] * DT)
    return np.array(x), np.array(v)


def loss(b: float) -> float:
    return (forward(b)[0][-1] - TARGET) ** 2


def manual_grad(b: float) -> tuple[float, np.ndarray]:
    """Backward pass by hand: walk the chain rule from x[N] back to b."""
    x, v = forward(b)
    dL_dx = 2 * (x[-1] - TARGET)  # local derivative of (x − target)²
    # dx[N]/dv[k+1] = dt for every step (x adds v[k+1]·dt), plus what v[k+1] passes on to later v's
    dxN_dv = np.zeros(N + 1)
    acc = 0.0  # dx[N]/dv[k+1], accumulated from the end
    dL_db = 0.0
    for k in reversed(range(N)):
        acc += DT  # x[k+1] = x[k] + v[k+1]·dt  →  ∂x/∂v[k+1] = dt
        dxN_dv[k + 1] = acc
        alive = 1.0 if v[k + 1] > 0 else 0.0  # max(0, ·) passes the gradient only while moving
        dL_db += dL_dx * acc * alive * (-DT)  # ∂v[k+1]/∂b = −dt (direct dependence)
        acc *= alive * (1 - DRAG * DT)  # ∂v[k+1]/∂v[k] = 1 − DRAG·dt
    return dL_db, dxN_dv


def torch_grad(b: float) -> float:
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    v, x = torch.tensor(V0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    for _ in range(N):
        v = torch.clamp(v - bt * DT - DRAG * v * DT, min=0.0)
        x = x + v * DT
    ((x - TARGET) ** 2).backward()
    return bt.grad.item()


def numeric_grad(b: float, h: float = 1e-4) -> float:
    return (loss(b + h) - loss(b - h)) / (2 * h)


def gradient_descent(b0=15.0, lr=0.06, steps=12) -> list[dict]:
    b, path = b0, []
    for k in range(steps):
        g, _ = manual_grad(b)
        row = {"step": k, "b": b, "stop_x": float(forward(b)[0][-1]), "L": loss(b), "manual": g, "numeric": numeric_grad(b), "torch": torch_grad(b)}
        path.append(row)
        print(f"step {k:2d}  b = {b:7.2f}  stops at {row['stop_x']:6.1f} (target {TARGET:.0f})  "
              f"dL/db: manual {g:+10.3f}  numeric {row['numeric']:+10.3f}  torch {row['torch']:+10.3f}")
        b -= lr * g
    return path


def plot(path) -> None:
    bs = np.linspace(5, 200, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(bs, [loss(b) for b in bs], color="#64748b", lw=2, label="L(b) = (stop position − target)²")
    a1.plot([p["b"] for p in path], [p["L"] for p in path], "o-", color="#2563eb", label="gradient descent on b")
    a1.set(xlabel="brake force b", ylabel="loss", yscale="log", title="Gradient from the chain rule → GD finds the right brake")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    b_final = path[-1]["b"]
    _, dxN_dv = manual_grad(b_final)
    stop_t = (forward(b_final)[1] > 0).sum() * DT
    t = np.arange(1, N + 1) * DT
    a2.bar(t, dxN_dv[1:], width=DT * 0.9, color="#2563eb")
    a2.axvline(stop_t, color="#dc2626", ls="--", lw=1)
    a2.text(stop_t + 0.03, dxN_dv.max() * 0.9, "car stopped:\nmax(0, ·) blocks\nthe gradient", color="#dc2626", fontsize=8)
    a2.set(xlabel="time of the step [s]", ylabel="∂x[N] / ∂v[k]",
           title=f"Final brake b = {b_final:.1f}: how much each moment's speed moves the stop point\n(one step back = one more factor (1 − drag·dt) in the chain)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/ch02b_chain_rule.png", dpi=150)


def animate(path, watch: bool) -> None:
    cv = Canvas(watch, "Ch. 02b · Chain rule")
    pg = cv.pg
    picks = [p for p in path if p["step"] in (0, 1, 3, len(path) - 1)]
    sx = lambda x: 60 + x * 1.4  # world px → screen px
    lanes = [150 + i * 110 for i in range(len(picks))]
    traj = [forward(p["b"])[0] for p in picks]
    for k in range(N + 25):
        i = min(k, N)
        cv.screen.fill((34, 77, 52))
        for p, y, xs, col in zip(picks, lanes, traj, COLORS):
            pg.draw.rect(cv.screen, (70, 72, 78), (40, y - 30, 820, 60))
            pg.draw.line(cv.screen, (255, 255, 255), (sx(TARGET), y - 30), (sx(TARGET), y + 30), 4)
            cv.car(np.array([sx(xs[i]), 640 - y]), 0.0, col, scale=1.6)
            cv.screen.blit(cv.font.render(f"GD step {p['step']}: b = {p['b']:.1f}", True, col), (50, y - 52))
        cv.screen.blit(cv.small.render("white line = target", True, MUTED), (sx(TARGET) - 50, 90))
        best = path[-1]
        cv.panel("02b · Chain rule", [
            "car brakes with force b,",
            f"must stop on the line ({TARGET:.0f} px)",
            "",
            "dL/db = dL/dx · dx/dv · dv/db",
            ("summed over all 90 time steps", MUTED),
            "",
            ("gradient at GD step 0:", MUTED),
            f"  by hand: {path[0]['manual']:+.2f}",
            f"  numeric: {path[0]['numeric']:+.2f}",
            f"  torch:   {path[0]['torch']:+.2f}",
            "",
            (f"after {best['step']} GD steps:", COLORS[3]),
            (f"  b = {best['b']:.1f}, stops at {best['stop_x']:.1f}", COLORS[3]),
        ])
        if not cv.show():
            break
    if not watch:
        cv.save_gif("assets/ch02b_chain_rule.gif", every=2)


def main(watch: bool) -> None:
    path = gradient_descent()
    json.dump({"v0": V0, "target": TARGET, "path": path}, open("results_ch02b.json", "w"), indent=2)
    plot(path)
    animate(path, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
