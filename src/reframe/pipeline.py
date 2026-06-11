"""High-level convenience wrappers that run the whole thing on a video file.
These need the `ml` extra (detectors). The lower-level `build_crop_path` stays
detector-agnostic and core-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .path import build_crop_path
from .presets import DEFAULT_PRESET, Preset, resolve_preset
from .types import CropPath

if TYPE_CHECKING:
    from .asd import ASDBackend


def analyze_video(
    video_path: str,
    preset: str | Preset = DEFAULT_PRESET,
    aspect: tuple[int, int] = (9, 16),
    use_face: bool = True,
    asd_backend: ASDBackend | None = None,
    overrides: dict | None = None,
) -> CropPath:
    """Video file -> crop path (detect + scene-detect + smooth). Needs `ml`.

    `preset` is a name or a Preset; `overrides` patches individual preset fields
    (e.g. {"max_step_x": 4.0, "deadzone": 0.06}) without editing presets.py.

    If `asd_backend` is given, the active-speaker cue comes from that model (run in
    its own env, merged in by frame) instead of the built-in lip heuristic — the
    lip mesh is skipped, the rest of the pipeline is identical.
    """
    from .detect import YoloDetector
    from .scenes import scene_starts

    p = resolve_preset(preset, overrides)
    starts = scene_starts(video_path)
    # with an ASD backend, the speaker's face box comes from the model (stable,
    # smoothed) — so skip MediaPipe entirely (no BlazeFace framing, no lip mesh).
    det = YoloDetector(
        video_path, classes=p.classes,
        use_face=use_face and asd_backend is None,
        use_lips=asd_backend is None,
    )
    frames = det.frames()

    if asd_backend is not None:
        from .asd import apply_speaking_scores

        scores = asd_backend.run(video_path)
        frames = apply_speaking_scores(frames, scores)

    return build_crop_path(det.meta, frames, scene_starts=starts, preset_name=p, target_aspect=aspect)


def reframe_video(
    input_path: str,
    output_path: str,
    preset: str | Preset = DEFAULT_PRESET,
    aspect: tuple[int, int] = (9, 16),
    use_face: bool = True,
    asd_backend: ASDBackend | None = None,
    overrides: dict | None = None,
    **render_kwargs,
) -> str:
    """Video file -> final reframed mp4 (analyze + render). Needs `ml` + ffmpeg."""
    from .render import render_video

    path = analyze_video(
        input_path, preset=preset, aspect=aspect, use_face=use_face,
        asd_backend=asd_backend, overrides=overrides,
    )
    return render_video(path, input_path, output_path, **render_kwargs)
