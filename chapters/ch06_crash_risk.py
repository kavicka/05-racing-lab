"""Chapter 06 · Predicting rare crashes: why ROC-AUC looks great and the alarm is still useless.

Step 06 showed on synthetic data that ROC-AUC hides class imbalance. Here the imbalance is real:
cars with a random `aggression` (some far too brave) and a little steering noise drive the tracks,
and for every frame we ask "will this car crash within the next second?". Crashes are rare frames.

1. Collect frames from 10 training tracks and 10 unseen test tracks. Label = crash within 30 frames (1 s).
2. A logistic regression on the 8 sensor values estimates P(crash in 1 s | sensors).
3. On the test tracks: ROC-AUC vs precision-recall, and how many false alarms per real one.
4. The GIF: cars on a test track, each with its crash risk; a ring = alarm. Red ring = it really
   crashed within 1 s, orange = false alarm.

Run:  uv run python -m chapters.ch06_crash_risk            (results_ch06.json + plot + GIF)
      uv run python -m chapters.ch06_crash_risk --watch
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from racing.car import Cars
from racing.drivers import Baseline
from racing.render import COLORS, MUTED, Canvas
from racing.track import TEST_SEEDS, TRAIN_SEEDS, generate

HORIZON = 30  # frames = 1 s
N_CARS, MAX_FRAMES = 40, 20 * 30


def features(obs: np.ndarray) -> np.ndarray:
    """8 raw sensors plus a few squared / product terms so a linear model can bend a little."""
    front, left, right, speed = obs[:, 3], obs[:, :3].min(1), obs[:, 4:7].min(1), obs[:, 7]
    return np.column_stack([obs, speed / (front + 0.05), speed**2, np.minimum(left, right)])


def drive(seed: int, rng: np.random.Generator, n: int = N_CARS):
    """Drive n noisy cars on one track; return per-frame sensors and the frame each car crashed (or −1)."""
    track = generate(seed)
    cars = Cars(track, n)
    aggr = rng.uniform(0.5, 1.7, n)
    driver = Baseline(aggression=aggr)
    obs_log, car_log, frame_log = [], [], []
    crash_frame = np.full(n, -1)
    for f in range(MAX_FRAMES):
        alive = cars.alive & ~cars.finished
        if not alive.any():
            break
        obs = cars.observe()
        idx = np.flatnonzero(alive)
        obs_log.append(obs[idx])
        car_log.append(idx)
        frame_log.append(np.full(len(idx), f))
        steer, throttle = driver(obs)
        steer = steer + rng.normal(0, 0.25, n)  # an imperfect driver
        was = cars.alive.copy()
        cars.step(steer, throttle)
        crash_frame[was & ~cars.alive] = f
    obs, car, frame = np.concatenate(obs_log), np.concatenate(car_log), np.concatenate(frame_log)
    cf = crash_frame[car]
    y = (cf >= 0) & (cf - frame < HORIZON)
    return obs, y, aggr


def dataset(seeds, rng):
    X, Y = [], []
    for s in seeds:
        o, y, _ = drive(s, rng)
        X.append(o)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def evaluate(model, X, y) -> dict:
    p = model.predict_proba(features(X))[:, 1]
    prec, rec, thr = precision_recall_curve(y, p)
    i = int(np.argmin(np.abs(rec[:-1] - 0.8)))
    fpr, tpr, _ = roc_curve(y, p)
    step = max(1, len(fpr) // 400)
    return {
        "frames": int(len(y)), "positive_frames": int(y.sum()), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "threshold_recall_80": float(thr[i]), "precision_at_recall_80": float(prec[i]),
        "false_alarms_per_true_alarm": float((1 - prec[i]) / prec[i]),
        "accuracy_always_safe": float(1 - y.mean()),
        "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
        "pr": {"recall": rec[::step].tolist(), "precision": prec[::step].tolist()},
    }


def plot(res) -> None:
    t = res["test"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(t["roc"]["fpr"], t["roc"]["tpr"], color="#2563eb", lw=2, label=f"ROC-AUC = {t['roc_auc']:.3f}")
    a1.plot([0, 1], [0, 1], "k:", lw=1)
    a1.set(xlabel="false positive rate", ylabel="recall (crashes caught)", title="ROC on unseen tracks: looks excellent")
    a2.plot(t["pr"]["recall"], t["pr"]["precision"], color="#dc2626", lw=2, label=f"PR-AUC = {t['pr_auc']:.3f}")
    a2.axhline(t["positive_rate"], color="#94a3b8", ls="--", lw=1, label=f"random guess = {t['positive_rate']:.1%} crash frames")
    a2.plot([0.8], [t["precision_at_recall_80"]], "o", color="k")
    a2.annotate(f"catch 80 % of crashes:\n{t['precision_at_recall_80']:.0%} of alarms are real\n"
                f"({t['false_alarms_per_true_alarm']:.1f} false per true)",
                (0.8, t["precision_at_recall_80"]), xytext=(-170, 40), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->"})
    a2.set(xlabel="recall (crashes caught)", ylabel="precision (alarms that are real)", ylim=(0, 1.02),
           title="Precision–recall: the honest picture")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=9, loc="lower right" if a is a1 else "upper right")
    fig.suptitle(f"Will this car crash within 1 s?  Only {t['positive_rate']:.1%} of frames are 'yes'", fontsize=12)
    fig.tight_layout()
    fig.savefig("assets/ch06_crash_risk.png", dpi=150)


def show(model, res, watch: bool) -> None:
    rng = np.random.default_rng(7)
    track = generate(TEST_SEEDS[3])
    n = 8
    cars = Cars(track, n)
    aggr = np.linspace(0.6, 1.7, n)
    driver = Baseline(aggression=aggr)
    thr = res["test"]["threshold_recall_80"]
    hist_pos = [[] for _ in range(n)]
    alarms = []  # (frame, car) of every alarm, judged later
    crash_frame = np.full(n, -1)
    frames = []
    cv = Canvas(watch, "Ch. 06 · Crash risk")
    pg = cv.pg
    counts = {"true": 0, "false": 0, "missed": 0}
    for f in range(int(16 * 30)):
        obs = cars.observe()
        risk = model.predict_proba(features(obs))[:, 1]
        active = cars.alive & ~cars.finished
        for i in np.flatnonzero(active & (risk > thr)):
            alarms.append((f, i))
        frames.append((cars.pos.copy(), cars.heading.copy(), risk.copy(), active.copy()))
        steer, throttle = driver(obs)
        was = cars.alive.copy()
        cars.step(steer + rng.normal(0, 0.25, n), throttle)
        crash_frame[was & ~cars.alive] = f
        if not (cars.alive & ~cars.finished).any():
            break
    # now that we know the future, judge each alarm and replay
    alarm_set = {}
    for f, i in alarms:
        real = crash_frame[i] >= 0 and 0 <= crash_frame[i] - f < HORIZON
        alarm_set[(f, i)] = real
    for f, (pos, head, risk, active) in enumerate(frames):
        cv.track(track)
        for i in range(n):
            if not active[i]:
                continue
            hist_pos[i].append(pos[i])
            col = COLORS[i % len(COLORS)]
            cv.car(pos[i], head[i], col, scale=1.1)
            if (f, i) in alarm_set:
                ring = (239, 68, 68) if alarm_set[(f, i)] else (249, 115, 22)
                c = (pos[i][0], 640 - pos[i][1])
                pg.draw.circle(cv.screen, ring, c, 18, 3)
        seen = [(ff, i) for (ff, i) in alarm_set if ff <= f]
        counts["true"] = sum(alarm_set[k] for k in seen)
        counts["false"] = len(seen) - counts["true"]
        t = res["test"]
        y = cv.panel("06 · Crash risk", [
            "P(crash within 1 s | sensors)",
            ("logistic regression, 8 sensors", MUTED),
            f"alarm when risk > {thr:.2f}",
            ("(threshold that catches 80 %)", MUTED),
            "",
            ("● real alarm (crashed in 1 s)", (239, 68, 68)),
            ("● false alarm", (249, 115, 22)),
            f"alarm frames: {counts['true']} real, {counts['false']} false",
            "",
            ("on 10 unseen tracks:", MUTED),
            f"crash frames: {t['positive_rate']:.1%}",
            f"ROC-AUC  {t['roc_auc']:.3f}",
            f"PR-AUC   {t['pr_auc']:.3f}",
            f"{t['false_alarms_per_true_alarm']:.1f} false alarms per real one",
        ])
        for i in range(n):
            if active[i]:
                cv.bar(cv.screen.get_width() - 314 + (i % 2) * 150, y + 18 + (i // 2) * 30, 140, frames[f][2][i], 0, 1,
                       COLORS[i % len(COLORS)], f"car {i + 1} (a={aggr[i]:.1f}) risk {frames[f][2][i]:.2f}")
        if not cv.show():
            break
    for _ in range(20):
        cv.show()
    if not watch:
        cv.save_gif("assets/ch06_crash_risk.gif")


def main(watch: bool) -> None:
    rng = np.random.default_rng(0)
    Xtr, ytr = dataset(TRAIN_SEEDS, rng)
    Xte, yte = dataset(TEST_SEEDS, rng)
    print(f"train: {len(ytr)} frames, {ytr.mean():.2%} crash-within-1s   test: {len(yte)} frames, {yte.mean():.2%}")
    model = LogisticRegression(max_iter=2000).fit(features(Xtr), ytr)
    res = {"horizon_frames": HORIZON, "train": evaluate(model, Xtr, ytr), "test": evaluate(model, Xte, yte)}
    for k in ("train", "test"):
        r = res[k]
        print(f"{k}: ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  precision@recall80 {r['precision_at_recall_80']:.3f}  "
              f"false/true {r['false_alarms_per_true_alarm']:.1f}  'always safe' accuracy {r['accuracy_always_safe']:.2%}")
    json.dump(res, open("results_ch06.json", "w"), indent=1)
    plot(res)
    show(model, res, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
