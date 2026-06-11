"""Preset resolution + field overrides — pure Python."""

import math

import pytest

from reframe import Preset, build_crop_path, resolve_preset
from reframe.detect import ReplayDetector
from reframe.types import Detection, FrameDetections, VideoMeta

W, H, FPS = 1920, 1080, 30.0


def test_resolve_by_name_is_the_bundled_preset():
    from reframe import PRESETS

    assert resolve_preset("talking_head") is PRESETS["talking_head"]


def test_overrides_patch_fields_and_keep_the_rest():
    p = resolve_preset("talking_head", {"max_step_x": 4.0, "deadzone": 0.06})
    assert p.max_step_x == 4.0
    assert p.deadzone == 0.06
    # untouched fields keep the preset's values; name is preserved
    assert p.name == "talking_head"
    assert p.max_zoom == resolve_preset("talking_head").max_zoom


def test_unknown_override_raises():
    # a typo must error, not silently no-op
    with pytest.raises(TypeError):
        resolve_preset("talking_head", {"max_step_xx": 4.0})


def test_accepts_a_preset_object():
    custom = resolve_preset("talking_head", {"max_step_x": 99.0})
    assert resolve_preset(custom).max_step_x == 99.0


def test_override_flows_through_build_crop_path():
    # a tiny pan cap should bound the per-frame step in the actual path
    frames = []
    for i in range(60):
        sx = 960 + 400 * math.sin(i / 60 * math.pi)
        frames.append(FrameDetections(i, [Detection("person", 0.9, sx - 150, 200, sx + 150, 900, track_id=1)]))
    meta = VideoMeta(fps=FPS, width=W, height=H, source="syn")
    det = ReplayDetector(meta, frames, track=False)
    preset = resolve_preset("talking_head", {"max_step_x": 2.0, "deadzone": 0.0})
    path = build_crop_path(meta, det.frames(), scene_starts=[0], preset_name=preset, target_aspect=(9, 16), keyframe_fps=FPS)
    for a, b in zip(path.keyframes, path.keyframes[1:]):
        assert abs(b.cx - a.cx) <= 2.0 + 1e-3
    assert path.preset == "talking_head"
