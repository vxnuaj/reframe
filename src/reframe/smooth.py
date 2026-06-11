"""The virtual camera: a confidence-weighted, velocity-damped follower.

This is where "cinematic vs nervous security-cam" is decided. Each axis (crop
center x, y, and zoom) is a damped spring:

    target  <- ease(target, observation)         # smooth the goal itself
    velocity <- velocity*damping + (target-value)*response
    velocity <- clamp(velocity, +/- max_step)     # cap pan speed
    value   <- clamp(value + velocity, bounds)    # integrate + keep in-frame

Low detection confidence shrinks both the target-easing and the max step, so a
shaky/uncertain detection nudges the camera gently instead of snapping to it.
"""

from __future__ import annotations

from dataclasses import dataclass


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class Follower:
    """One damped-spring axis."""

    value: float
    response: float
    damping: float
    target_alpha: float
    target: float = 0.0
    velocity: float = 0.0

    def __post_init__(self) -> None:
        self.target = self.value

    def update(
        self,
        observation: float,
        confidence: float,
        max_step: float,
        lo: float,
        hi: float,
        deadzone: float = 0.0,
        response_scale: float = 1.0,
    ) -> float:
        # confidence in [0,1]; floor it so the camera never fully freezes.
        c = clamp(confidence, 0.15, 1.0)
        # deadzone: while the subject stays within this band of where the camera
        # already is, don't chase it — slight body sway shouldn't pan the frame.
        if abs(observation - self.value) > deadzone:
            alpha = clamp(self.target_alpha * (0.55 + c * 0.65), 0.04, 0.4)
            self.target = clamp(lerp(self.target, observation, alpha), lo, hi)
        # advance the value toward the target with a capped, damped velocity.
        # response_scale > 1 briefly during a subject switch = a quick eased whip.
        step_cap = max(1.0, max_step * (0.35 + c * 0.65))
        self.velocity = self.velocity * self.damping + (self.target - self.value) * (self.response * response_scale)
        self.velocity = clamp(self.velocity, -step_cap, step_cap)
        self.value = clamp(self.value + self.velocity, lo, hi)
        return self.value

    def snap(self, value: float) -> None:
        """Hard cut (e.g. a scene boundary): jump with no residual velocity."""
        self.value = value
        self.target = value
        self.velocity = 0.0


def base_crop_size(frame_w: int, frame_h: int, aspect_w: int, aspect_h: int) -> tuple[int, int]:
    """The largest crop of the target aspect that fits the frame (zoom = 1)."""
    target = aspect_w / aspect_h
    if frame_w / frame_h >= target:  # source wider than target -> full height
        h = frame_h
        w = round(h * target)
    else:  # source taller -> full width
        w = frame_w
        h = round(w / target)
    return min(w, frame_w), min(h, frame_h)


class Camera:
    """Drives crop center + zoom from per-frame observations."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        aspect: tuple[int, int],
        preset,
        init_cx: float,
        init_cy: float,
        init_zoom: float = 1.0,
    ) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.preset = preset
        self.base_w, self.base_h = base_crop_size(frame_w, frame_h, aspect[0], aspect[1])
        self.x = Follower(init_cx, preset.motion_response, preset.motion_damping, preset.target_alpha)
        self.y = Follower(init_cy, preset.motion_response, preset.motion_damping, preset.target_alpha)
        self.zoom = Follower(init_zoom, preset.zoom_response, preset.zoom_damping, preset.zoom_alpha)
        self._boost = 0  # frames of fast-ease remaining (set on a subject switch)
        self._boost_factor = 1.0

    def crop_size(self, zoom: float) -> tuple[int, int]:
        w = clamp(round(self.base_w / zoom), 16, self.frame_w)
        h = clamp(round(self.base_h / zoom), 16, self.frame_h)
        return int(w), int(h)

    def update(self, obs_cx: float, obs_cy: float, obs_zoom: float, confidence: float) -> tuple[float, float, float]:
        p = self.preset
        boosting = self._boost > 0
        scale = self._boost_factor if boosting else 1.0
        if boosting:
            self._boost -= 1
        z = self.zoom.update(clamp(obs_zoom, p.min_zoom, p.max_zoom), confidence, 1.0, p.min_zoom, p.max_zoom)
        cw, ch = self.crop_size(z)
        # the center must keep the crop window fully inside the frame
        dzx, dzy = p.deadzone * self.frame_w, p.deadzone * self.frame_h
        cx = self.x.update(obs_cx, confidence, p.max_step_x * scale, cw / 2, self.frame_w - cw / 2,
                           deadzone=dzx, response_scale=scale)
        cy = self.y.update(obs_cy, confidence, p.max_step_y * scale, ch / 2, self.frame_h - ch / 2,
                           deadzone=dzy, response_scale=scale)
        return cx, cy, z

    def boost(self, frames: int, factor: float = 3.0) -> None:
        """Briefly speed up the center followers so a subject switch eases across
        quickly (a whip) instead of snapping or crawling. Scene cuts still snap()."""
        self._boost = frames
        self._boost_factor = factor

    def snap_to(self, cx: float, cy: float, zoom: float) -> None:
        cw, ch = self.crop_size(zoom)
        self.zoom.snap(clamp(zoom, self.preset.min_zoom, self.preset.max_zoom))
        self.x.snap(clamp(cx, cw / 2, self.frame_w - cw / 2))
        self.y.snap(clamp(cy, ch / 2, self.frame_h - ch / 2))
