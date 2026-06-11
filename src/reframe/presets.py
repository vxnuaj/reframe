"""Per-content-type tuning. Two groups of knobs:
  - framing limits: which classes count as subjects, zoom range, max pan speed.
  - camera feel: the damped-spring response/damping/target-easing (see smooth.py).

Only `talking_head` ships today, and it's the one that's actually been tuned and
tested. To add another content type (e.g. sports, pets), define a new Preset and
register it in PRESETS below; it then works everywhere by name (CLI `--preset`,
`analyze_video(preset=...)`). Different content wants different feel: looser, faster
tracking and a wider zoom range for fast motion; tight, gentle framing for a head.
"""

from __future__ import annotations

import dataclasses
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
    # NOTE: add more presets here as new content types are tuned (e.g. "sports",
    # "pets", "cars"). A starting point usually wants a looser zoom range, higher
    # max_step_x/y for fast motion, and a smaller deadzone. Validate on real footage
    # before trusting it — the values are a feel, not a default that just works.
}

DEFAULT_PRESET = "talking_head"


def resolve_preset(preset: str | Preset = DEFAULT_PRESET, overrides: dict | None = None) -> Preset:
    """Resolve a preset by name (or take a Preset as-is), then optionally override
    individual fields without editing presets.py:

        resolve_preset("talking_head", {"max_step_x": 4.0, "deadzone": 0.06})

    Unknown field names raise TypeError, so a typo is caught rather than ignored.
    """
    base = preset if isinstance(preset, Preset) else PRESETS[preset]
    if overrides:
        base = dataclasses.replace(base, **overrides)
    return base
