"""Chapter 11 · Driving from pixels: a CNN replaces the ray sensors, Grad-CAM shows where it looks.

1. Camera: instead of 7 ray distances the car gets a 64×64 top-down picture of the road ahead,
   rotated so that "up" is always the car's heading. Speed is still given as one number.
2. Training: imitation with DAgger as in chapter 04, now in PyTorch on MPS. Two students get the same
   recipe: a CNN and an MLP on the flattened pixels with a similar number of weights.
3. Grad-CAM: which part of the picture pushed the steering decision right now?
4. Both students drive 10 tracks they have never seen.

Run:  uv run python -m chapters.ch11_pixels            (trains, writes results and GIF)
      uv run python -m chapters.ch11_pixels --watch    (live window, reuses saved weights)
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from racing.car import DT, Cars  # noqa: E402
from racing.drivers import Baseline  # noqa: E402
from racing.render import COLORS, MUTED, Canvas  # noqa: E402
from racing.track import TEST_SEEDS, TRAIN_SEEDS, WORLD_H, WORLD_W, Track, generate  # noqa: E402

CAM = 64  # pixels per side
CAM_SCALE = 5.0  # world px per camera pixel: the picture covers 320 × 320 world px
CAR_ROW = 50  # the car sits low in the picture, because the road ahead matters more than behind
ROUNDS, EPOCHS_FIRST, EPOCHS_ROUND = 5, 30, 12
SEEDS = (0, 1, 2)
UNSEEN_SEEDS = TEST_SEEDS + list(range(200, 240))  # 10 tracks per lap are too few to tell two students apart
TEACHER = Baseline()
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
WEIGHTS = Path("data/ch11_pixels.pt")


def track_map(track: Track) -> np.ndarray:
    """Rasterise a track once: grass 0, asphalt 0.5, kerb 1. Array is indexed [y, x] with y up."""
    im = Image.new("L", (WORLD_W, WORLD_H), 0)
    d = ImageDraw.Draw(im)
    d.polygon([tuple(p) for p in track.right], fill=128)
    d.polygon([tuple(p) for p in track.left], fill=0)
    for edge in (track.left, track.right):
        d.line([tuple(p) for p in edge] + [tuple(edge[0])], fill=255, width=6)  # wider than one camera pixel
    return np.asarray(im, dtype=np.float32) / 255


def camera(world: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
    """Sample the map along the car's own axes. Every picture looks the same way up: egocentric view."""
    rows, cols = np.mgrid[0:CAM, 0:CAM]
    forward = (CAR_ROW - rows) * CAM_SCALE
    left = (CAM / 2 - cols) * CAM_SCALE
    c, s = np.cos(heading), np.sin(heading)
    x = pos[0] + forward * c - left * s
    y = pos[1] + forward * s + left * c
    xi, yi = x.round().astype(int), y.round().astype(int)
    inside = (xi >= 0) & (xi < WORLD_W) & (yi >= 0) & (yi < WORLD_H)
    img = np.zeros((CAM, CAM), np.float32)
    img[inside] = world[yi[inside], xi[inside]]
    return img


class PixelCNN(nn.Module):
    """Shrinks 64×64 to 8×8. Flattens instead of global average pooling: for steering it matters
    WHERE the road is, while a classifier only cares THAT the object is there."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU()
        )
        self.last = nn.Conv2d(32, 32, 3, 1, 1)  # Grad-CAM reads this layer
        self.head = nn.Sequential(nn.Linear(32 * 8 * 8 + 1, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, img: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.last(self.conv(img)))
        return self.head(torch.cat([z.flatten(1), speed], 1))


class PixelMLP(nn.Module):
    """The same pixels without convolution: every pixel gets its own weights."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CAM * CAM + 1, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, img: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([img.flatten(1), speed], 1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Student:
    """Runs a trained network on the CPU: one frame at a time is faster there than on the GPU."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def act(self, img: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(img)[None, None]
        out = self.model(x, torch.tensor([[obs[0, 7]]], dtype=torch.float32))[0].numpy()
        return np.clip(out[:1], -1, 1), np.clip(out[1:], -1, 1)


def teacher_act(img, obs) -> tuple[np.ndarray, np.ndarray]:
    return TEACHER(obs)


def drive(act, seed: int, record: list | None = None, max_time: float = 40.0) -> dict:
    track = generate(seed)
    world = track_map(track)
    cars = Cars(track, 1)
    for _ in range(int(max_time / DT)):
        if cars.finished[0] or not cars.alive[0]:
            break
        obs = cars.observe()
        img = camera(world, cars.pos[0], cars.heading[0])
        if record is not None:
            record.append((img, obs[0]))
        cars.step(*act(img, obs))
    return {
        "finished": bool(cars.finished[0]),
        "lap_time": float(cars.lap_time[0]),
        "progress": float(min(cars.progress[0] / track.length, 1.0)),
    }


def evaluate(act, seeds) -> dict:
    res = [drive(act, s) for s in seeds]
    done = [r for r in res if r["finished"]]
    return {
        "finished": len(done),
        "tracks": len(res),
        "mean_lap": float(np.mean([r["lap_time"] for r in done])) if done else None,
        "mean_progress": float(np.mean([r["progress"] for r in res])),
    }


def collect(act, seeds=TRAIN_SEEDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whoever drives, the labels always come from the teacher looking at the same moment."""
    frames: list = []
    for s in seeds:
        drive(act, s, frames)
    imgs = np.stack([f[0] for f in frames])[:, None]
    obs = np.stack([f[1] for f in frames])
    steer, throttle = TEACHER(obs)
    labels = np.stack([np.clip(steer, -1, 1), np.clip(throttle, -1, 1)], 1).astype(np.float32)
    return imgs, obs[:, 7:8].astype(np.float32), labels


def train(model: nn.Module, data, epochs: int, losses: list, batch: int = 128, lr: float = 1e-3) -> None:
    model.to(DEVICE).train()
    imgs, speed, y = (torch.from_numpy(d).to(DEVICE) for d in data)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(y) - batch + 1, batch):
            idx = order[i : i + batch]
            x, s, t = imgs[idx], speed[idx], y[idx].clone()
            # all tracks run counter-clockwise, so without mirroring the student rarely sees a right turn
            flip = torch.rand(len(idx), device=DEVICE) < 0.5
            x[flip] = x[flip].flip(3)
            t[flip, 0] = -t[flip, 0]
            loss = F.mse_loss(model(x, s), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    model.cpu().eval()


def train_student(model: nn.Module, name: str) -> tuple[Student, dict]:
    data = collect(teacher_act)
    losses: list[float] = []
    train(model, data, EPOCHS_FIRST, losses)
    student = Student(model)
    rounds = [{"round": 0, "examples": len(data[0]), "start_step": 0, "test": evaluate(student.act, TEST_SEEDS)}]
    print(name, rounds[-1], flush=True)
    for r in range(1, ROUNDS + 1):
        new = collect(student.act)
        data = tuple(np.concatenate(pair) for pair in zip(data, new))
        start = len(losses)
        train(model, data, EPOCHS_ROUND, losses)
        rounds.append({"round": r, "examples": len(data[0]), "start_step": start, "test": evaluate(student.act, TEST_SEEDS)})
        print(name, rounds[-1], flush=True)
    return student, {"params": count_params(model), "rounds": rounds, "losses": losses[::5]}


def grad_cam(model: PixelCNN, img: np.ndarray, speed: float) -> tuple[np.ndarray, float]:
    """Which 8×8 cells push the current steering command? Uses |steer|, so left and right turns look alike.

    Classic Grad-CAM averages the gradient over the map, which is right only for a global-average-pool
    head. This head flattens, so the same channel can mean opposite things in different cells, and the
    average points at nonsense. Multiplying gradient and activation cell by cell (HiResCAM) stays faithful.
    """
    maps = {}
    handle = model.last.register_forward_hook(lambda m, i, out: maps.update(a=out))
    x = torch.from_numpy(img)[None, None]
    steer = model(x, torch.tensor([[speed]], dtype=torch.float32))[0, 0]
    grads = torch.autograd.grad(steer.abs(), maps["a"])[0]
    handle.remove()
    cam = F.relu((grads * maps["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(CAM, CAM), mode="bilinear", align_corners=False)[0, 0].detach()
    return (cam / (cam.max() + 1e-8)).numpy(), float(steer.item())


def cam_on_kerbs(model: PixelCNN, seeds=TEST_SEEDS, every: int = 15) -> dict:
    """Is the heatmap on the road edges? Compare its share there with the area those pixels take up."""
    frames: list = []
    for s in seeds:
        drive(teacher_act, s, frames)
    mass, area = [], []
    for img, obs in frames[::every]:
        cam, _ = grad_cam(model, img, float(obs[7]))
        kerb = F.max_pool2d(torch.from_numpy(img > 0.8).float()[None, None], 5, 1, 2)[0, 0].numpy() > 0  # within 2 px
        if cam.sum() > 0:
            mass.append(cam[kerb].sum() / cam.sum())
            area.append(kerb.mean())
    return {"frames": len(mass), "cam_share_near_kerbs": float(np.mean(mass)), "area_share_near_kerbs": float(np.mean(area))}


def heat_rgb(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Grey camera picture with the heatmap blended on top in orange."""
    grey = np.repeat((img[..., None] * 200).astype(np.float32), 3, 2)
    hot = cam[..., None] ** 1.5
    return (grey * (1 - hot) + np.array([249, 115, 22], np.float32) * hot).astype(np.uint8)


def plot(results: dict, cnn: PixelCNN) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    for key, color, label in (("cnn", "#2563eb", "CNN"), ("mlp", "#dc2626", "MLP on pixels")):
        losses = np.array(results[key][0]["losses"])
        w = 25
        a1.plot(np.arange(w - 1, len(losses)) * 5, np.convolve(losses, np.ones(w) / w, "valid"), color=color, lw=1.5, label=label)
    a1.set(xlabel="training step", ylabel="MSE vs teacher", title="Training (DAgger adds harder data)", yscale="log")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    for key, color, label in (("cnn", "#2563eb", "CNN"), ("mlp", "#dc2626", "MLP on pixels")):
        per_seed = np.array([[r["test"]["finished"] for r in run["rounds"]] for run in results[key]])
        a2.plot(np.arange(ROUNDS + 1), per_seed.T, color=color, lw=0.8, alpha=0.4)
        a2.plot(np.arange(ROUNDS + 1), per_seed.mean(0), "-o", color=color, label=f"{label}, mean of {len(SEEDS)} seeds")
    a2.axhline(results["teacher"]["test"]["finished"], color="#94a3b8", ls="--", label="teacher (rays)")
    a2.set(xlabel="DAgger round", ylabel="unseen tracks finished", ylim=(-0.5, 10.5), title="Laps on the 10 test tracks")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=9)
    frames: list = []
    drive(teacher_act, TEST_SEEDS[0], frames)
    img, obs = frames[len(frames) // 3]
    cam, steer = grad_cam(cnn, img, float(obs[7]))
    a3.imshow(heat_rgb(img, cam))
    a3.set(title=f"Grad-CAM, steer = {steer:+.2f}", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig("assets/ch11_pixels.png", dpi=150)


def show(cnn: PixelCNN, results: dict, watch: bool) -> None:
    student = Student(cnn)
    seed = next((s for s in TEST_SEEDS if drive(student.act, s)["finished"]), TEST_SEEDS[0])
    track = generate(seed)
    world = track_map(track)
    cars = Cars(track, 1)
    cv = Canvas(watch, "Ch. 11 · Driving from pixels")
    trail = []
    for _ in range(int(14 * 30)):
        obs = cars.observe()
        img = camera(world, cars.pos[0], cars.heading[0])
        steer, throttle = student.act(img, obs)
        cam, _ = grad_cam(cnn, img, float(obs[0, 7]))
        cars.step(steer, throttle)
        trail.append(cars.pos[0].copy())
        cv.track(track)
        cv.path(trail[-150:], (148, 163, 184), 2)
        cv.car(cars.pos[0], cars.heading[0], COLORS[4], scale=1.4)
        state = f"lap {cars.lap_time[0]:.2f} s ✓" if cars.finished[0] else ("crashed" if not cars.alive[0] else f"{cars.time[0]:.1f} s")
        y = cv.panel("11 · Driving from pixels", [
            f"CNN, {results['cnn'][0]['params']:,} weights",
            ("no rays: only pixels + speed", MUTED),
            ("track never seen in training", MUTED),
            f"{state}   speed {cars.speed[0]:4.0f} px/s",
        ])
        x0 = cv.screen.get_width() - 330
        cv.image(x0 + 37, y + 26, heat_rgb(img, cam), 4, "what it sees + Grad-CAM (orange = decisive)")
        cv.bar(x0 + 16, y + 310, 140, float(steer[0]), -1, 1, COLORS[0], f"steer {steer[0]:+.2f}")
        cv.bar(x0 + 172, y + 310, 140, float(throttle[0]), -1, 1, COLORS[2], f"gas {throttle[0]:+.2f}")
        if not cv.show() or cars.finished[0] or not cars.alive[0]:
            break
    for _ in range(20):
        cv.show()
    if not watch:
        cv.save_gif("assets/ch11_pixels.gif")


def experiment(make, name: str) -> tuple[list[dict], nn.Module]:
    """Same recipe for every seed. The first seed's model goes on, so no model is picked by its test score."""
    runs, first = [], None
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make()
        student, res = train_student(model, f"{name} seed {seed}")
        res.update(seed=seed, train=evaluate(student.act, TRAIN_SEEDS), test=evaluate(student.act, TEST_SEEDS))
        res["unseen"] = evaluate(student.act, UNSEEN_SEEDS)
        print(name, seed, "train", res["train"], "test", res["test"], "unseen", res["unseen"], flush=True)
        runs.append(res)
        first = first or model
    return runs, first


def main(watch: bool) -> None:
    if watch and WEIGHTS.exists():
        cnn = PixelCNN()
        cnn.load_state_dict(torch.load(WEIGHTS))
        cnn.eval()
        show(cnn, json.load(open("results_ch11.json")), watch)
        return
    print("device:", DEVICE, "weights: cnn", count_params(PixelCNN()), "mlp", count_params(PixelMLP()))
    res_cnn, cnn = experiment(PixelCNN, "cnn")
    torch.save(cnn.state_dict(), WEIGHTS)
    res_mlp, _ = experiment(PixelMLP, "mlp")
    teacher = {"train": evaluate(teacher_act, TRAIN_SEEDS), "test": evaluate(teacher_act, TEST_SEEDS)}
    teacher["unseen"] = evaluate(teacher_act, UNSEEN_SEEDS)
    print("teacher", teacher)
    results = {"device": str(DEVICE), "cnn": res_cnn, "mlp": res_mlp, "teacher": teacher, "gradcam": cam_on_kerbs(cnn)}
    print(results["gradcam"])
    json.dump(results, open("results_ch11.json", "w"), indent=1)
    plot(results, cnn)
    show(cnn, results, watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    main(ap.parse_args().watch)
