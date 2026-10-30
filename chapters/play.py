"""Drive yourself with the arrow keys and record your laps as training data for chapter 04.

←/→ steer, ↑ gas, ↓ brake, N = next track, Esc = quit and save.
Every frame stores (8 sensor values, [steer, throttle]) into data/my_laps.npz.

Run:  uv run python -m chapters.play
      uv run python -m chapters.ch04_imitation --demos data/my_laps.npz
"""

from pathlib import Path

import numpy as np

from racing.car import Cars, ray_directions
from racing.render import COLORS, MUTED, Canvas
from racing.track import TRAIN_SEEDS, generate


def main() -> None:
    cv = Canvas(True, "Racing Lab · you drive")
    pg = cv.pg
    obs_log, act_log = [], []
    t_i, laps = 0, []
    track = generate(TRAIN_SEEDS[t_i])
    cars = Cars(track, 1)
    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                running = False
            elif e.type == pg.KEYDOWN and e.key == pg.K_n:
                t_i = (t_i + 1) % len(TRAIN_SEEDS)
                track, cars = generate(TRAIN_SEEDS[t_i]), None
        if cars is None:
            cars = Cars(track, 1)
        keys = pg.key.get_pressed()
        steer = float(keys[pg.K_LEFT]) - float(keys[pg.K_RIGHT])
        throttle = float(keys[pg.K_UP]) - float(keys[pg.K_DOWN])
        obs = cars.observe()
        obs_log.append(obs[0])
        act_log.append([steer, throttle])
        cars.step(np.array([steer]), np.array([throttle]))
        if cars.finished[0]:
            laps.append(cars.lap_time[0])
        if cars.finished[0] or not cars.alive[0]:
            cars = Cars(track, 1)

        cv.track(track)
        dists = obs[0, :7] * 220
        cv.rays(cars.pos[0], ray_directions(cars.heading[:1])[0], dists)
        cv.car(cars.pos[0], cars.heading[0], COLORS[4], scale=1.3)
        cv.panel("You drive", [
            f"track {TRAIN_SEEDS[t_i]}   time {cars.time[0]:5.1f} s",
            f"speed {cars.speed[0]:4.0f} px/s",
            f"recorded frames: {len(obs_log)}",
            f"finished laps: {len(laps)}" + (f", best {min(laps):.2f} s" if laps else ""),
            "",
            ("←/→ steer  ↑ gas  ↓ brake", MUTED),
            ("N next track   Esc save & quit", MUTED),
        ])
        pg.display.flip()
        cv.clock.tick(30)

    if obs_log:
        Path("data").mkdir(exist_ok=True)
        np.savez("data/my_laps.npz", obs=np.array(obs_log), act=np.array(act_log, dtype=float))
        print(f"saved {len(obs_log)} frames to data/my_laps.npz")


if __name__ == "__main__":
    main()
