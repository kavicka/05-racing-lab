# Tutorial · 05 · Racing Lab

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

The Racing Lab is different from the other steps: it is a small simulator that grows a new **chapter** whenever a roadmap step has a concept that maps naturally to cars. This tutorial first builds the shared simulator step by step (track → car → sensors → driver → simulation loop → rendering), then explains how each chapter uses it and why it is built that way.

**Goal:** a world simple enough that you can understand every number the car sees (8 inputs) and every number it outputs (2 controls), so that each ML idea can be watched, not just measured.

**You need:** projects 01–04 (NumPy, matrices, autograd). Later chapters use PyTorch and scikit-learn.

---

## Part A · The simulator ([`racing/`](racing/))

### Step 1 · Generate a track ([`racing/track.py`](racing/track.py))

```python
def generate(seed, n_points=360, width=60.0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = np.ones_like(theta)
    for k in (2, 3, 4, 5, 6):
        amp = rng.uniform(0.04, 0.3 / k**0.5)
        r += amp * np.cos(k * theta + rng.uniform(0, 2*np.pi))
    center = [W/2 + rx * r * cos(theta), H/2 + ry * r * sin(theta)]
```

**How it works:** start from a circle (radius 1) and add a few wobbles: `cos(k·θ)` with k = 2…6 gives 2 to 6 bumps around the loop, each with a random size and phase. The sum is a smooth closed curve, the **centerline**.

**Why this way:**
- **Always closed and smooth**, because `cos(kθ)` repeats every 2π for whole-number k.
- **Doesn't cross itself**, because the amplitudes are small (`< 0.3/√k`) and the radius stays positive.
- **A seed = a track.** `TRAIN_SEEDS = 0–9` and `TEST_SEEDS = 100–109` give 10 training tracks and 10 tracks the drivers **never see during training**. That train/test split is what lets later chapters ask "did it learn to drive, or did it memorise 10 tracks?".

**Walls from the centerline** (`Track.__post_init__`):
```python
tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)       # direction along the track
normal  = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)    # tangent rotated +90°
self.left  = c + normal * width/2
self.right = c - normal * width/2
```
Rotating a 2-D vector `(x, y)` by 90° gives `(−y, x)`; that is the normal. Shifting the centerline left and right along it gives the two walls, stored as short line segments (for ray casting).

### Step 2 · Car physics for many cars at once ([`racing/car.py`](racing/car.py))

```python
self.speed   = clip(speed + (throttle*ACCEL - DRAG*speed) * DT, 0, MAX_SPEED)
self.heading = heading + steer * TURN_RATE * DT
vel          = [cos(heading), sin(heading)] * speed
self.pos     = pos + vel * DT
```

**How it works:** arcade physics with three state variables per car (position, heading, speed) and two controls in [−1, 1]: `steer` (positive = left) and `throttle` (negative = brake). `DT = 1/30 s`, one frame. Drag (`−DRAG·speed`) gives a natural top speed.

**Why vectorised:** every attribute is a NumPy array of shape `(N,)`, so one `step()` moves all N cars. Chapter 12 runs 60 cars; chapter 06 runs 40 per track. A Python loop over cars would be the slow path from project 03.

**Why `np.where(live, new, old)`:** crashed or finished cars must freeze while the rest keep going, without breaking the vectorisation.

**Progress along the track** (`_update_progress`): each car tracks the nearest centerline point, searching only near its last index (`-10 … +30`). That is cheap and can never "jump" to the other side of the track. Progress is the sum of the centerline segments passed (negative if driving backwards). If the car is farther than `width/2` from the centerline, it left the road: `alive = False`.

### Step 3 · Sensors: 7 rays as 2×2 linear systems

```python
RAY_ANGLES = deg2rad([-90, -60, -30, 0, 30, 60, 90])

def cast_rays(pos, heading, walls):
    d  = ray_directions(heading)           # (N, R, 2) = R(heading + offset) @ [1, 0]
    ...
    det = cross(dd, -e)                    # (N, R, W)
    t   = cross(ap, -e) / det              # distance along the ray
    u   = cross(dd, ap) / det              # position along the wall segment
    hit = (abs(det) > 1e-9) & (t > 0) & (0 <= u <= 1)
```

**How it works:** a ray is `p + t·d`, a wall segment is `a + u·(b − a)`. They meet where both are equal: two equations (x and y), two unknowns (t, u). That is a 2×2 linear system, solved with Cramer's rule (ratio of determinants, written as 2-D cross products). A hit counts only if `t > 0` (in front of the car) and `0 ≤ u ≤ 1` (on the segment). `det = 0` means the ray is parallel to the wall.

**Why this way:** it is project 03's linear algebra doing real work, and with broadcasting all **cars × rays × walls** are solved in one shot, shape `(N, R, W)`. The minimum `t` over walls is the distance to the nearest wall.

**The observation** (`Cars.observe`): 7 distances / 220 and speed / max speed → **8 numbers in [0, 1]**. Scaling to similar ranges matters: a neural network learns badly when one input is 400 and another is 0.1.

### Step 4 · A hand-written baseline driver ([`racing/drivers.py`](racing/drivers.py))

```python
left  = rays[:, 4:].mean(1)            # +30°, +60°, +90°
right = rays[:, :3].mean(1)            # −90°, −60°, −30°
steer = steer_gain * (left - right)    # steer toward the side with more room
target = clip(rays[:, 3] * aggression * 1.4, 0.25, 1.0)   # faster when the road ahead is long
throttle = clip((target - speed) * 5, -1, 1)
```

**Why a baseline:** every chapter needs something to compare against, and some need a **teacher** to copy. This rule-based driver finishes 9 of 10 test tracks. It has two knobs (`steer_gain`, `aggression`) that chapter 02 tunes with gradient descent.

**Interface used everywhere:** a driver is any callable `obs (N, 8) → (steer (N,), throttle (N,))`. The baseline, the micrograd MLP, the CNN and the PPO policy all follow it, so they can all be dropped into the same simulation and evaluation code.

### Step 5 · The simulation loop ([`racing/sim.py`](racing/sim.py))

```python
def run(driver, track, n=1, max_time=60.0, on_step=None):
    cars = Cars(track, n)
    for _ in range(int(max_time / DT)):
        if not (cars.alive & ~cars.finished).any(): break
        obs = cars.observe()
        steer, throttle = driver(obs)
        cars.step(steer, throttle)
        if on_step: on_step(cars, obs, steer, throttle)
```

The classic **observe → act → step** loop, the same loop reinforcement learning uses. `on_step` is a hook so chapters can record data (e.g. sensor frames for training) without changing the loop. `lap()` wraps it for one car and returns finished / lap time / progress.

### Step 6 · Rendering ([`racing/render.py`](racing/render.py))

pygame draws the track, cars and rays, plus a side panel for numbers. The same `Canvas` either opens a live window (`--watch`) or, headless (`SDL_VIDEODRIVER=dummy`), saves frames for the GIFs in `assets/`. Only the renderer flips the y-axis; the physics uses math coordinates (y up), so the linear algebra stays clean.

---

## Part B · The chapters

Each chapter is one script in [`chapters/`](chapters/) that writes `results_chNN.json`, a plot and a GIF.

### Ch. 01 · Batching 200 brains ([`ch01_batch.py`](chapters/ch01_batch.py))

`MLPBatch` gives each of N cars its own random MLP 8 → 16 → 2 and evaluates them two ways:
```python
loop:    for i in range(N): h = tanh(obs[i] @ W1[i] + b1[i]) ...
batched: h = tanh((obs[:, None] @ W1)[:, 0] + b1)    # (N,1,8) @ (N,8,16) → (N,1,16)
```
The `None` adds a dimension so each car's observation becomes a 1×8 matrix; `@` then multiplies all N pairs at once (project 03's batching rule). Same output (`allclose`), **21× faster** at 200 cars. Project 01's lesson, made visible.

### Ch. 02 · Chain rule and gradient descent ([`ch02b_chain_rule.py`](chapters/ch02b_chain_rule.py), [`ch02_gradient.py`](chapters/ch02_gradient.py))

- **Part A:** a car brakes with force `b` and must stop at 250 px. Its stop point depends on `b` through 90 physics steps, so `dL/db` is a 90-link chain rule. Computed by hand (a backward loop over time), numerically and with PyTorch: all give −650.934. Gradient descent finds `b = 66.5`. When the car stops, `max(0, ·)` has derivative 0 and blocks the gradient, like a ReLU.
- **Part B:** the loss is the mean lap time over the training tracks as a function of `aggression`. There is no formula, so the gradient is estimated numerically, and GD lowers lap time 18.2 → 13.1 s. **Why a shrinking learning rate:** near the "crash cliff" the loss is not smooth, and a fixed step kept jumping across it.

### Ch. 03 · Vectors and matrices ([`ch03_vectors.py`](chapters/ch03_vectors.py))

Pure visualisation of project 03: heading = rotation matrix × [1, 0], dot product with the track direction = alignment, rays = linear systems from Step 3.

### Ch. 04 · Imitation learning with my micrograd ([`ch04_imitation.py`](chapters/ch04_imitation.py))

1. **Behaviour cloning:** the baseline drives the training tracks; every 3rd frame becomes `(8 sensors → teacher's steer, throttle)`. An MLP 8 → 8 → 2 built from **project 04's `Value` class** trains with MSE.
2. **DAgger:** the *student* drives, the teacher labels each state it visits, the new examples are added, retrain (3 rounds).

**Why DAgger:** the teacher never makes mistakes, so cloning data never shows "too close to the wall, what now?". The student drifts into states the teacher never visited and errors snowball. DAgger collects exactly those states. Unseen tracks finished: 5/10 → **7/10**.

### Ch. 06 · Predicting rare crashes ([`ch06_crash_risk.py`](chapters/ch06_crash_risk.py))

Cars with random aggression and noisy steering drive; every frame is labelled *"crashes within the next 30 frames (1 s)"*. Only 6.3 % of frames are positive, so the data is naturally imbalanced. A logistic regression on the 8 sensors plus a few extra features (`speed/front`, `speed²`, `min(left, right)`) predicts the risk.

**Why the extra features:** logistic regression draws a straight boundary; squared and ratio terms let it bend a little without switching to a bigger model.

**Lesson:** ROC-AUC 0.910 looks excellent, but at 80 % recall the precision is only 0.25: three false alarms per real one. Project 06's point on real data.

### Ch. 11 · Driving from pixels ([`ch11_pixels.py`](chapters/ch11_pixels.py))

- **Camera:** each track is rasterised once (grass 0, asphalt 0.5, kerb 1). `camera()` samples a 64×64 window **in the car's own frame** (rotated so "up" is always forward). Egocentric images mean "turn left" always looks the same, wherever the car is.
- **Models:** a CNN and an MLP on flattened pixels with a similar number of weights, trained with DAgger on MPS. Mirroring images and flipping steering doubles the data (all tracks are counter-clockwise, so right turns are rare otherwise).
- **Why no global average pooling in the CNN:** a classifier asks *whether* something is in the image; a driver needs to know *where* the road is. Averaging over positions would throw that away, so the head flattens.
- **Grad-CAM bug kept as a lesson:** classic Grad-CAM assumes a pooled head. With a flattening head, HiResCAM (gradient × activation cell by cell) is the correct variant.
- **3 seeds per model:** CNN 38.0 vs MLP 34.7 of 50 unseen tracks on average, but the ranges overlap, so it is called a trend, not a proof.

### Ch. 12 · Reinforcement learning with PPO ([`ch12_ppo.py`](chapters/ch12_ppo.py))

No teacher at all. The car only gets a **reward**: progress since the last frame (scaled) minus 1 for a crash.

- **`Fleet`:** 60 cars on 10 tracks that never stop. A crashed car **respawns at a random point** (otherwise it would only ever practise the first corner). Half of respawns are **standing starts**, added after finding cars that spun round at speed 0 and drove the wrong way.
- **`ActorCritic`:** two separate MLPs 8 → 64 → 64. The **actor** outputs the mean of a Gaussian over (steer, throttle) plus a learned `log_std`, so sampling adds exploration noise. The **critic** predicts future reward from a state. Separate networks so the critic's large errors don't disturb the actor. The actor's last layer starts tiny so every action starts near 0.
- **Collect:** 200 steps with the current noisy policy, storing observations, actions, their log-probabilities, rewards and critic values.
- **GAE (advantage):**
  ```python
  delta = r[t] + gamma * V[t+1] * alive - V[t]     # one-step surprise
  running = delta + gamma * lam * alive * running   # weighted sum of future surprises
  ```
  The advantage says *how much better this action turned out than the critic expected*. `gamma = 0.99` discounts the far future; `lam = 0.95` trades bias vs noise.
- **PPO update:**
  ```python
  ratio = exp(new_logp - old_logp)
  policy_loss = -min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
  ```
  Make good actions more likely, bad ones less, but the clip removes any reward for moving the policy more than ±20 % away from the one that collected the data, because that data says nothing reliable about a very different policy.
- **The clock is not a crash:** when a car is stopped after 40 s it still had a future, so its last reward gets `gamma × V(last state)` added. A crash really ends it (future = 0).

**Result:** two of three seeds finish all 50 unseen tracks and lap ~34 % faster than the baseline; one seed gets stuck in a worse habit (43/50). The copying control matches its teacher exactly, because copying cannot exceed its teacher. **Lesson:** RL can surpass the teacher, but one run is not a reliable result; report several seeds.

---

## Run it

```bash
uv sync
uv run python -m chapters.ch01_batch --watch
uv run python -m chapters.ch04_imitation      # needs 04-autograd-makemore cloned next to this repo
uv run python -m chapters.ch12_ppo --watch    # reuses saved weights
uv run python -m chapters.play                # drive yourself with the arrow keys
```

## Try it yourself

1. Add a track generator option with clockwise tracks and see how the ch. 12 policy copes (it has only ever turned mostly left).
2. Change `RAY_ANGLES` to 5 rays. Does the baseline still finish 9/10?
3. Record your own laps with `chapters.play` and use them as the teacher in ch. 04 (`--demos data/my_laps.npz`).

## Recap

| Piece | Idea |
|---|---|
| Track | Circle + random harmonics = smooth closed loop; seeds give train vs unseen test tracks |
| Car | Vectorised arcade physics; `np.where` freezes dead cars |
| Sensors | Ray × wall = 2×2 linear system, solved for all cars × rays × walls at once |
| Driver | One interface `(N, 8) → steer, throttle` for rules, MLPs, CNNs and RL |
| Chapters | Each roadmap concept (batching, chain rule, imitation, imbalance, CNN, RL) shown on the same world |
