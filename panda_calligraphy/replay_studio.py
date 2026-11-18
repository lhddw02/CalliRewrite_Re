"""Display-only ink settings and playback state for a fixed verified trajectory.

The native mode uses the existing deposit function without changing footprints.
Studio mode is an explicitly artistic preview, never a verification result.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


NATIVE_MODE = "Native faithful"
STUDIO_MODE = "Ink studio"


@dataclass(frozen=True)
class ReplaySettings:
    mode: str = NATIVE_MODE
    opacity: float = 1.0
    ink_level: float = 100.0
    absorbency: float = 0.0
    ink_color: tuple = (18, 18, 20)
    footprint_scale: float = 1.0
    stiffness: float = 1.0
    splay: float = 1.0
    guide_grid: bool = False
    fibers: bool = True
    loop: bool = False
    speed: float = 1.0
    show_arm: bool = True

    def __post_init__(self):
        if self.mode not in (NATIVE_MODE, STUDIO_MODE):
            raise ValueError("Unknown replay display mode")
        for name, lo, hi in (("opacity",0,1), ("ink_level",0,100),
                             ("absorbency",0,3), ("footprint_scale",.25,2.5),
                             ("stiffness",.3,3), ("splay",.25,2.5), ("speed",.25,4)):
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be finite and within [{lo}, {hi}]")
        color = tuple(self.ink_color)
        if len(color) != 3 or any(not isinstance(c, (int, np.integer)) or not 0 <= c <= 255 for c in color):
            raise ValueError("ink_color must contain three integer RGB channels")
        object.__setattr__(self, "ink_color", color)


class PreviewCanvas:
    """Incremental sample raster; rebuild only when footprint style/window changes."""
    def __init__(self, replay):
        self.replay = replay
        self.mask = np.zeros((replay.resolution,)*2, bool)
        self.cursor = 0
        self.key = None

    def update(self, end_frame, settings, start_frame=0):
        from callirewrite_replay import mask_rgb, stamp
        end_frame = min(len(self.replay.time_s)-1, int(end_frame))
        start_frame = max(0, int(start_frame))
        scale = settings.footprint_scale if settings.mode == STUDIO_MODE else 1.0
        key = (settings.mode, scale, start_frame)
        if key != self.key or end_frame < self.cursor-1:
            self.mask[:] = False
            self.cursor, self.key = start_frame, key
        while self.cursor <= end_frame:
            if settings.mode == NATIVE_MODE:
                self.replay.deposit(self.mask, self.cursor)
            else:
                footprint = self.replay.footprint(self.cursor)
                if footprint is not None:
                    uv, radius, points = footprint
                    if points is not None:
                        points = uv + (points-uv)*scale
                    # Studio intentionally uses a pixel-scaled round footprint
                    # together with the exported triangle, for a smoother mark.
                    # The official normalized-radius artifact remains native-only.
                    stamp(self.mask, uv, radius*scale, points, native=False)
            self.cursor += 1
        if settings.mode == NATIVE_MODE:
            return mask_rgb(self.mask), self.mask.copy()
        alpha = self.mask.astype(float)
        if settings.absorbency:
            # A soft halo, not a fluid or porous-paper model. Units are pixels
            # at 300px canvas width, scaled with the raster resolution.
            halo = gaussian_filter(alpha, settings.absorbency*self.replay.resolution/300)
            alpha = np.maximum(alpha, .7*halo)
        alpha *= settings.opacity*settings.ink_level/100
        paper = np.array((251,247,238), dtype=float)
        color = np.asarray(settings.ink_color, dtype=float)
        rgb = np.rint(paper+(color-paper)*alpha[...,None]).clip(0,255).astype(np.uint8)
        return rgb, self.mask.copy()


def render_preview(replay, end_frame, settings=ReplaySettings(), start_frame=0):
    return PreviewCanvas(replay).update(end_frame, settings, start_frame)


class ReplayPlayback:
    """Owned by one display loop; callbacks send commands through a queue."""
    def __init__(self, replay, autoplay=False):
        self.replay = replay
        self.time = 0.0
        self.playing = bool(autoplay)
        self.paint_start = 0
        self.step_end = None

    @property
    def frame(self):
        return max(0, min(len(self.replay.time_s)-1,
            int(np.searchsorted(self.replay.time_s, self.time, side="right")-1)))

    def restart(self, playing=False):
        self.time, self.paint_start, self.step_end = 0.0, 0, None
        self.playing = bool(playing)

    def clear(self):
        self.paint_start = self.frame+1

    def toggle(self):
        self.step_end = None
        if self.time >= self.replay.time_s[-1]:
            self.restart(playing=True)
        else:
            self.playing = not self.playing

    def step_stroke(self):
        candidates = np.flatnonzero((self.replay.contact_index >= 0) &
                                    (self.replay.time_s > self.time+1e-10))
        if not len(candidates):
            self.playing = False
            return False
        first = candidates[0]
        stroke = self.replay.source["stroke_id"][self.replay.contact_index[first]]
        contacts = np.flatnonzero(self.replay.contact_index >= 0)
        same = contacts[self.replay.source["stroke_id"][self.replay.contact_index[contacts]] == stroke]
        self.step_end = float(self.replay.time_s[same[-1]])
        self.playing = True
        return True

    def advance(self, dt, settings):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Playback delta must be finite and nonnegative")
        if not self.playing:
            return None
        end = float(self.replay.time_s[-1])
        limit = end if self.step_end is None else min(end, self.step_end)
        self.time = min(limit, self.time+dt*settings.speed)
        if self.time >= limit:
            self.playing = False
            self.step_end = None
            return "completed" if self.time >= end else "stroke"
        return None


def save_preview_settings(path, settings, playback):
    """Store display choices separately from immutable verification artifacts."""
    path = Path(path)
    data = {"schema":"callirewrite.preview-settings.v1", "scope":"Display-only preview; source trajectory and verification remain unchanged",
            "settings":asdict(settings), "time_s":float(playback.time),
            "frame":playback.frame, "paint_start_frame":playback.paint_start,
            "source_sha256":playback.replay.source.get("metadata",{}).get("source_sha256"),
            "ink_semantics":"Ink studio settings restyle all visible accumulated contacts, using pixel-scaled round footprints plus the exported triangles. Ink level is uniform, not consumed over time; Dip ink sets it to 100%."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+"\n")
    return data
