"""Chapter 04 · A neural network on YOUR autograd engine learns to drive.

1. Demonstrations: a teacher drives laps; we record (8 sensor values → steer, throttle).
   The teacher is the baseline driver, or you (`python -m chapters.play` records your laps).
2. Training: an MLP 8 → 8 → 2 built from `Value` objects of step 04 (micrograd) learns to copy
   the teacher with mean-squared error, backward() and gradient descent. No PyTorch.
3. DAgger: the student drives, the teacher says what it *would* have done in the student's
   situations, those examples are added, retrain. This fixes "small mistakes snowball".
4. The student drives on 10 tracks it has never seen, with its neurons lighting up live.

Run:  uv run python -m chapters.ch04_imitation                       (teacher = baseline)
      uv run python -m chapters.ch04_imitation --demos data/my_laps.npz   (teacher = you)
      uv run python -m chapters.ch04_imitation --watch
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-autograd-makemore"))  # reuse step 04's engine
from micrograd.engine import Value  # noqa: E402
from micrograd.nn import MLP  # noqa: E402

from racing.car import Cars  # noqa: E402
from racing.drivers import Baseline  # noqa: E402
from racing.render import COLORS, MUTED, Canvas  # noqa: E402
from racing.sim import lap, run  # noqa: E402
from racing.track import TEST_SEEDS, TRAIN_SEEDS, generate  # noqa: E402

TEACHER = Baseline()
IN_LABELS = ["−90°", "−60°", "−30°", "0°", "+30°", "+60°", "+90°", "speed"]


class Student:
    """Wraps the micrograd MLP so the simulator can call it on (N, 8) arrays."""

    def __init__(self, hidden: int = 8, seed: int = 0):
        random.seed(seed)
        self.net = MLP(8, [hidden, 2])

    def forward_values(self, x) -> tuple[list, list]:
        hidden = self.net.layers[0]([Value(v) for v in x])
        out = self.net.layers[1](hidden)
        return hidden, out

    def __call__(self, obs: np.ndarray):
        acts = np.array([[o.data for o in self.forward_values(x)[1]] for x in obs])
        return np.clip(acts[:, 0], -1, 1), np.clip(acts[:, 1], -1, 1)

    def activations(self, x) -> list[np.ndarray]:
        h, o = self.forward_values(x)
        return [np.asarray(x) * 2 - 1, np.array([v.data for v in h]), np.clip([v.data for v in o], -1, 1)]


def teacher_labels(obs: np.ndarray) -> np.ndarray:
    s, t = TEACHER(obs)
    return np.stack([np.clip(s, -1, 1), np.clip(t, -1, 1)], 1)


def collect(driver, seeds, every: int = 3) -> np.ndarray:
    """Drive `driver` on the tracks and record observations (labels come from the teacher)."""
    X = []
    for s in seeds:
        run(driver, generate(s), 1, 40.0, on_step=lambda cars, obs, *_: X.append(obs[0]))
    return np.array(X[::every])


def train(student: Student, X, Y, steps: int, lr: float = 0.05, batch: int = 32, losses=None) -> list[float]:
    losses = [] if losses is None else losses
    params = student.net.parameters()
    for k in range(steps):
        idx = np.random.randint(0, len(X), batch)
        loss = Value(0.0)
        for i in idx:
            out = student.net([Value(v) for v in X[i]])
            loss = loss + (out[0] - Y[i, 0]) ** 2 + (out[1] - Y[i, 1]) ** 2  # squared error
        loss = loss / batch
        student.net.zero_grad()
        loss.backward()  # step 04's backward()
        for p in params:
            p.data -= lr * p.grad
        losses.append(loss.data)
    return losses


def evaluate(driver, seeds) -> dict:
    res = [lap(driver, generate(s), 40.0) for s in seeds]
    done = [r for r in res if r.finished]
    return {"finished": len(done), "tracks": len(res), "mean_lap": float(np.mean([r.lap_time for r in done])) if done else None,
            "mean_progress": float(np.mean([r.progress for r in res]))}


def plot(losses, rounds, table) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(losses, color="#94a3b8", lw=0.8, label="minibatch loss")
    w = 25
    a1.plot(np.arange(w - 1, len(losses)), np.convolve(losses, np.ones(w) / w, "valid"), color="#2563eb", lw=2, label=f"moving average ({w})")
    for r in rounds[1:]:
        a1.axvline(r["start_step"], color="#dc2626", ls="--", lw=1)
    a1.set(xlabel="training step", ylabel="MSE vs teacher", title="Training on micrograd (red = new DAgger data added)", yscale="log")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    names = [t["driver"] for t in table]
    x = np.arange(len(names))
    a2.bar(x - 0.2, [t["train"]["finished"] for t in table], 0.4, label="training tracks (10)", color="#93c5fd")
    a2.bar(x + 0.2, [t["test"]["finished"] for t in table], 0.4, label="unseen test tracks (10)", color="#2563eb")
    a2.set_xticks(x, names, fontsize=9)
    a2.set(ylabel="laps finished", ylim=(0, 10.5), title="Who can finish a lap?")
    a2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/ch04_imitation.png", dpi=150)


def show(student: Student, losses, watch: bool) -> None:
    # show a full lap: the first unseen test track the student finishes
    seed = next((s for s in TEST_SEEDS if lap(student, generate(s), 40.0).finished), TEST_SEEDS[0])
    track = generate(seed)
    cars = Cars(track, 1)
    cv = Canvas(watch, "Ch. 04 · Imitation learning on micrograd")
    trail = []
    for _ in range(int(14 * 30)):
        obs = cars.observe()
        steer, throttle = student(obs)
        cars.step(steer, throttle)
        trail.append(cars.pos[0].copy())
        cv.track(track)
        cv.path(trail[-150:], (148, 163, 184), 2)
        cv.car(cars.pos[0], cars.heading[0], COLORS[3], scale=1.4)
        state = f"lap {cars.lap_time[0]:.2f} s ✓" if cars.finished[0] else ("crashed" if not cars.alive[0] else f"{cars.time[0]:.1f} s")
        y = cv.panel("04 · Learning to drive", [
            "MLP 8 → 8 → 2 on my micrograd",
            ("track never seen in training", MUTED),
            f"{state}   speed {cars.speed[0]:4.0f} px/s",
        ])
        x0 = cv.screen.get_width() - 330
        cv.network(x0 + 55, y + 10, 210, 240, student.activations(obs[0]), IN_LABELS, ["steer", "gas"])
        cv.bar(x0 + 16, y + 280, 140, steer[0], -1, 1, COLORS[0], f"steer {steer[0]:+.2f}")
        cv.bar(x0 + 172, y + 280, 140, throttle[0], -1, 1, COLORS[2], f"gas {throttle[0]:+.2f}")
        cv.chart(x0 + 16, y + 305, 298, 80, np.log10(np.convolve(losses, np.ones(25) / 25, "valid")), COLORS[0], "training loss (log10)")
        if not cv.show() or cars.finished[0] or not cars.alive[0]:
            break
    for _ in range(20):
        cv.show()
    if not watch:
        cv.save_gif("assets/ch04_imitation.gif")


def main(watch: bool, demos: str | None, dagger_rounds: int = 3) -> None:
    np.random.seed(0)
    student = Student()
    if demos:
        d = np.load(demos)
        X, Y = d["obs"], d["act"]
        print(f"loaded {len(X)} of your demonstration frames")
        dagger_rounds = 0  # you are not here to label the student's mistakes
    else:
        X = collect(TEACHER, TRAIN_SEEDS)
        Y = teacher_labels(X)
    print(f"round 0: {len(X)} examples from the teacher")
    losses, rounds = [], [{"round": 0, "examples": len(X), "start_step": 0}]
    train(student, X, Y, 600, losses=losses)
    cloning = evaluate(student, TRAIN_SEEDS), evaluate(student, TEST_SEEDS)
    print(f"behaviour cloning: train {cloning[0]}  test {cloning[1]}")

    for r in range(1, dagger_rounds + 1):
        Xs = collect(student, TRAIN_SEEDS)  # where does the STUDENT end up?
        X, Y = np.concatenate([X, Xs]), np.concatenate([Y, teacher_labels(Xs)])  # teacher labels those states
        rounds.append({"round": r, "examples": len(X), "start_step": len(losses)})
        train(student, X, Y, 300, losses=losses)
        print(f"DAgger round {r}: {len(X)} examples, train {evaluate(student, TRAIN_SEEDS)}")

    table = [
        {"driver": "baseline\n(teacher)", "train": evaluate(TEACHER, TRAIN_SEEDS), "test": evaluate(TEACHER, TEST_SEEDS)},
        {"driver": "MLP, cloning\nonly", "train": cloning[0], "test": cloning[1]},
        {"driver": f"MLP + DAgger\n({dagger_rounds} rounds)", "train": evaluate(student, TRAIN_SEEDS), "test": evaluate(student, TEST_SEEDS)},
    ]
    for t in table:
        print(t["driver"].replace("\n", " "), "| train", t["train"], "| test", t["test"])
    json.dump({"rounds": rounds, "losses": losses, "table": table}, open("results_ch04.json", "w"), indent=2)
    plot(losses, rounds, table)
    show(student, losses, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--demos", default=None, help="npz recorded with `python -m chapters.play`")
    a = ap.parse_args()
    main(a.watch, a.demos)
