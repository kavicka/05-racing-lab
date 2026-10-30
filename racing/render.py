"""Drawing with pygame. The same code draws into a live window (--watch) or into GIF frames."""

import os

import numpy as np

from racing.track import WORLD_H, WORLD_W, Track

PANEL_W = 330
SIZE = (WORLD_W + PANEL_W, WORLD_H)

GRASS = (34, 77, 52)
ASPHALT = (70, 72, 78)
KERB = (230, 230, 230)
TEXT = (235, 235, 235)
MUTED = (160, 165, 175)
PANEL = (24, 26, 32)
RAY = (250, 204, 21)
COLORS = [(59, 130, 246), (239, 68, 68), (34, 197, 94), (168, 85, 247), (249, 115, 22), (14, 165, 233)]


def _pg():
    import pygame  # imported lazily so headless scripts don't need a display

    return pygame


def to_screen(p: np.ndarray) -> np.ndarray:
    """Math coordinates (y up) → screen coordinates (y down)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], WORLD_H - p[..., 1]], -1)


class Canvas:
    def __init__(self, watch: bool = False, title: str = "Racing Lab", fps: int = 30):
        if not watch:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pg = _pg()
        pg.init()
        self.pg, self.watch, self.fps = pg, watch, fps
        self.screen = pg.display.set_mode(SIZE) if watch else pg.Surface(SIZE)
        if watch:
            pg.display.set_caption(title)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("dejavusans,arial", 16)
        self.big = pg.font.SysFont("dejavusans,arial", 22, bold=True)
        self.small = pg.font.SysFont("dejavusans,arial", 13)
        self.frames = []
        self._track_cache = None

    # --- world --------------------------------------------------------------
    def track(self, track: Track) -> None:
        if self._track_cache is None or self._track_cache[0] is not track:
            pg = self.pg
            surf = pg.Surface((WORLD_W, WORLD_H))
            surf.fill(GRASS)
            pg.draw.polygon(surf, ASPHALT, to_screen(track.right).tolist())  # outer edge
            pg.draw.polygon(surf, GRASS, to_screen(track.left).tolist())  # inner edge
            for edge in (track.left, track.right):
                pg.draw.lines(surf, KERB, True, to_screen(edge).tolist(), 2)
            pg.draw.line(surf, (255, 255, 255), *to_screen([track.left[0], track.right[0]]).tolist(), 4)
            self._track_cache = (track, surf)
        self.screen.blit(self._track_cache[1], (0, 0))

    def car(self, pos, heading, color=COLORS[0], alpha: bool = False, scale: float = 1.0) -> None:
        L, W = 22 * scale, 11 * scale
        corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        pts = to_screen(corners @ R.T + pos).tolist()
        if alpha:
            s = self.pg.Surface(SIZE, self.pg.SRCALPHA)
            self.pg.draw.polygon(s, (*color, 110), pts)
            self.screen.blit(s, (0, 0))
        else:
            self.pg.draw.polygon(self.screen, color, pts)
            nose = to_screen(R @ np.array([L / 2, 0]) + pos)
            self.pg.draw.circle(self.screen, (255, 255, 255), nose.tolist(), 2.5 * scale)

    def rays(self, pos, dirs, dists, color=RAY) -> None:
        for d, t in zip(dirs, dists):
            end = pos + d * t
            self.pg.draw.line(self.screen, color, to_screen(pos).tolist(), to_screen(end).tolist(), 1)
            self.pg.draw.circle(self.screen, color, to_screen(end).tolist(), 3)

    def arrow(self, start, vec, color, width: int = 3, label: str | None = None) -> None:
        pg = self.pg
        s, e = to_screen(start), to_screen(np.asarray(start) + vec)
        pg.draw.line(self.screen, color, s.tolist(), e.tolist(), width)
        d = e - s
        n = np.linalg.norm(d)
        if n > 1e-6:
            d = d / n
            perp = np.array([-d[1], d[0]])
            head = [e.tolist(), (e - d * 10 + perp * 5).tolist(), (e - d * 10 - perp * 5).tolist()]
            pg.draw.polygon(self.screen, color, head)
        if label:
            self.screen.blit(self.small.render(label, True, color), (e + 4).tolist())

    def path(self, pts, color, width: int = 2) -> None:
        if len(pts) > 1:
            self.pg.draw.lines(self.screen, color, False, to_screen(np.asarray(pts)).tolist(), width)

    # --- side panel -----------------------------------------------------------
    def panel(self, title: str, lines: list[str | tuple[str, tuple]], y0: int = 18) -> int:
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, (WORLD_W, 0, PANEL_W, WORLD_H))
        self.screen.blit(self.big.render(title, True, TEXT), (WORLD_W + 16, y0))
        y = y0 + 38
        for line in lines:
            text, color = (line, TEXT) if isinstance(line, str) else line
            self.screen.blit(self.font.render(text, True, color), (WORLD_W + 16, y))
            y += 22
        return y

    def bar(self, x, y, w, value, lo, hi, color, label) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (55, 58, 66), (x, y, w, 12))
        frac = (np.clip(value, lo, hi) - lo) / (hi - lo)
        if lo < 0 < hi:
            zero = x + w * (-lo) / (hi - lo)
            x0, x1 = sorted([zero, x + w * frac])
            pg.draw.rect(self.screen, color, (x0, y, max(1, x1 - x0), 12))
        else:
            pg.draw.rect(self.screen, color, (x, y, w * frac, 12))
        self.screen.blit(self.small.render(label, True, MUTED), (x, y - 15))

    def chart(self, x, y, w, h, values, color, label: str, ymin=None, ymax=None) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (34, 37, 44), (x, y, w, h))
        self.screen.blit(self.small.render(label, True, MUTED), (x + 4, y + 2))
        v = np.asarray(values, float)
        if len(v) < 2:
            return
        lo = np.nanmin(v) if ymin is None else ymin
        hi = np.nanmax(v) if ymax is None else ymax
        hi = hi if hi > lo else lo + 1
        xs = x + np.linspace(0, w, len(v))
        ys = y + h - 4 - (v - lo) / (hi - lo) * (h - 22)
        pg.draw.lines(self.screen, color, False, np.stack([xs, ys], 1).tolist(), 2)

    def network(self, x, y, w, h, layers: list[np.ndarray], labels_in=None, labels_out=None) -> None:
        """Draw activations of each layer as circles: blue = negative, orange = positive."""
        pg = self.pg
        n_layers = len(layers)
        cols = [x + 20 + i * (w - 40) / (n_layers - 1) for i in range(n_layers)]
        pos = []
        for c, a in zip(cols, layers):
            k = len(a)
            ys = [y + h / 2] if k == 1 else list(y + 12 + np.arange(k) * (h - 24) / (k - 1))
            pos.append([(c, yy) for yy in ys])
        for i in range(n_layers - 1):
            for p in pos[i]:
                for q in pos[i + 1]:
                    pg.draw.line(self.screen, (52, 55, 63), p, q, 1)
        for li, (pts, a) in enumerate(zip(pos, layers)):
            for j, (p, v) in enumerate(zip(pts, a)):
                v = float(np.clip(v, -1, 1))
                col = (int(60 + 190 * max(v, 0)), int(70 + 90 * abs(v)), int(60 + 190 * max(-v, 0)))
                pg.draw.circle(self.screen, col, p, 7)
                if li == 0 and labels_in:
                    self.screen.blit(self.small.render(labels_in[j], True, MUTED), (p[0] - 18 - 6 * len(labels_in[j]), p[1] - 8))
                if li == n_layers - 1 and labels_out:
                    self.screen.blit(self.small.render(labels_out[j], True, MUTED), (p[0] + 12, p[1] - 8))

    # --- output ---------------------------------------------------------------
    def show(self) -> bool:
        """Push a frame to the window or GIF buffer. Returns False when the user closes the window."""
        if self.watch:
            for e in self.pg.event.get():
                if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                    return False
            self.pg.display.flip()
            self.clock.tick(self.fps)
        else:
            self.frames.append(self.pg.image.tobytes(self.screen, "RGB"))
        return True

    def save_gif(self, path: str, every: int = 3, scale: float = 0.6, fps: int = 10) -> None:
        if not self.frames:
            return
        from PIL import Image

        imgs = []
        for raw in self.frames[::every]:
            im = Image.frombytes("RGB", SIZE, raw)
            im = im.resize((int(SIZE[0] * scale), int(SIZE[1] * scale)), Image.LANCZOS)
            imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
        self.frames = []

    def snapshot(self, path: str) -> None:
        self.pg.image.save(self.screen, path)
