"""Per-content-type tuning. Two groups of knobs:
  - framing limits: which classes count as subjects, zoom range, max pan speed.
  - camera feel: the damped-spring response/damping/target-easing (see smooth.py).

Values are tuned starting points — a talking head wants tight, gentle framing;
sports/cars want looser, faster tracking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    classes: tuple[str, ...]
    min_zoom: float
    max_zoom: float
    # max crop-center movement per frame (px @ source resolution, pre-confidence scaling)
    max_step_x: float
    max_step_y: float
    # camera spring (center)
    motion_response: float  # spring stiffness toward target
    motion_damping: float   # velocity retention (higher = smoother/heavier)
    target_alpha: float     # how fast the *target* eases toward the raw detection
    # camera spring (zoom)
    zoom_response: float
    zoom_damping: float
    zoom_alpha: float
    # deadzone: fraction of frame size the subject can move within before the
    # camera tracks at all (kills jitter from slight body sway). 0 = always track.
    deadzone: float = 0.0
    # switch_boost: frames of fast-ease when the subject changes (a quick whip to
    # the new speaker). 0 = hard snap on switch.
    switch_boost: int = 0


PRESETS: dict[str, Preset] = {
    "talking_head": Preset(
        name="talking_head",
        classes=("person",),
        min_zoom=1.05,
        max_zoom=1.85,
        max_step_x=4.0,
        max_step_y=3.0,
        motion_response=0.055,
        motion_damping=0.88,
        target_alpha=0.07,
        zoom_response=0.08,
        zoom_damping=0.85,
        zoom_alpha=0.035,
        deadzone=0.06,
        switch_boost=0,  # instant cut to the new speaker (true 1-frame cut)
    ),
    "sports": Preset(
        name="sports",
        classes=("person", "car", "bicycle", "motorcycle", "sports ball"),
        min_zoom=1.0,
        max_zoom=1.35,
        max_step_x=12.0,
        max_step_y=8.0,
        motion_response=0.18,
        motion_damping=0.78,
        target_alpha=0.2,
        zoom_response=0.1,
        zoom_damping=0.82,
        zoom_alpha=0.05,
    ),
    "pets": Preset(
        name="pets",
        classes=("dog", "cat", "person"),
        min_zoom=1.0,
        max_zoom=1.55,
        max_step_x=10.0,
        max_step_y=7.0,
        motion_response=0.15,
        motion_damping=0.8,
        target_alpha=0.16,
        zoom_response=0.09,
        zoom_damping=0.83,
        zoom_alpha=0.045,
    ),
    "cars": Preset(
        name="cars",
        classes=("car", "truck", "bus", "motorcycle", "person"),
        min_zoom=1.0,
        max_zoom=1.3,
        max_step_x=11.0,
        max_step_y=7.0,
        motion_response=0.17,
        motion_damping=0.79,
        target_alpha=0.18,
        zoom_response=0.1,
        zoom_damping=0.82,
        zoom_alpha=0.05,
    ),
}

DEFAULT_PRESET = "talking_head"
