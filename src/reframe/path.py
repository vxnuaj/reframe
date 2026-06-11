"""Turn a stream of per-frame detections into a smooth CropPath.

This is the decide step's output — pure data, no pixels. A renderer (our Remotion
service) interpolates between the keyframes and pans/zooms the source to match.
"""

from __future__ import annotations

from collections.abc import Iterable

from .presets import DEFAULT_PRESET, PRESETS, Preset
from .rank import SubjectSelector
from .smooth import Camera, base_crop_size, clamp
from .types import CropKeyframe, CropPath, Detection, FrameDetections, VideoMeta


def _focus(d: Detection, preset: Preset, base_h: int, fill_h: float) -> tuple[float, float, float]:
    """Where the camera *wants* to be for this subject: its (face) center, plus a
    desired zoom that makes it fill ~fill_h of the crop height."""
    fx = d.face_cx if (d.has_face and d.face_cx is not None) else d.cx
    fy = d.face_cy if (d.has_face and d.face_cy is not None) else d.cy
    desired_zoom = (fill_h * base_h) / max(d.h, 1.0)
    fz = clamp(desired_zoom, preset.min_zoom, preset.max_zoom)
    return fx, fy, fz


def build_crop_path(
    meta: VideoMeta,
    frames: Iterable[FrameDetections],
    scene_starts: Iterable[int] | None = None,
    preset_name: str = DEFAULT_PRESET,
    target_aspect: tuple[int, int] = (9, 16),
    keyframe_fps: float = 12.0,
    fill_h: float = 0.7,
    selector: SubjectSelector | None = None,
) -> CropPath:
    # preset_name may be a name or an already-resolved (possibly overridden) Preset
    preset = preset_name if isinstance(preset_name, Preset) else PRESETS[preset_name]
    selector = selector or SubjectSelector()
    starts = set(scene_starts or [0])
    starts.add(0)  # frame 0 is always a fresh shot

    base_w, base_h = base_crop_size(meta.width, meta.height, target_aspect[0], target_aspect[1])
    camera = Camera(meta.width, meta.height, target_aspect, preset, meta.width / 2, meta.height / 2, preset.min_zoom)

    stride = max(1, round(meta.fps / max(keyframe_fps, 1.0)))
    keyframes: list[CropKeyframe] = []
    last_obs: tuple[float, float, float] | None = None
    last_index = -1
    last_cxyz: tuple[float, float, float] | None = None  # previous frame's camera

    def emit(t: float, cxyz: tuple[float, float, float]) -> None:
        # keep keyframe times strictly increasing (a forced cut keyframe may land on
        # a frame a stride keyframe already covers)
        if keyframes and t <= keyframes[-1].t:
            return
        keyframes.append(CropKeyframe(t=t, cx=cxyz[0], cy=cxyz[1], zoom=cxyz[2]))

    for fd in frames:
        i = fd.frame_index
        last_index = i
        if i in starts:
            selector.reset()

        subject = selector.select(fd.detections, meta.width, meta.height)
        if subject is not None:
            obs = _focus(subject, preset, base_h, fill_h)
            conf = clamp(subject.conf, 0.2, 1.0)
            last_obs = obs
        elif last_obs is not None:
            obs = last_obs  # graceful hold: keep the last framing, barely moving
            conf = 0.15
        else:
            obs = (meta.width / 2, meta.height / 2, preset.min_zoom)
            conf = 0.15

        # Scene cut, or a subject switch when the preset wants a hard cut: snap. To
        # make it a TRUE cut (not a fast pan the renderer smears across the keyframe
        # gap), pin the old position one frame before and the new position at the cut
        # — so interpolation spans a single frame. switch_boost > 0 keeps the whip.
        # A cut needs BOTH a real turn change (selector.last_switch — not the id churn
        # the continuity bridge absorbs) AND a real reframe (the new subject is far
        # from where the camera is). A "switch" to a near-colocated track (a split of
        # the same person) moves nowhere — glide it, don't glitch-cut.
        real_switch = selector.last_switch and (
            last_cxyz is None or abs(obs[0] - last_cxyz[0]) > 0.08 * meta.width
        )
        is_cut = (i in starts) or (real_switch and preset.switch_boost <= 0)
        if is_cut:
            if last_cxyz is not None and i > 0:
                emit((i - 1) / meta.fps, last_cxyz)
            camera.snap_to(*obs)
            cx, cy, z = camera.x.value, camera.y.value, camera.zoom.value
            emit(i / meta.fps, (cx, cy, z))
        else:
            if real_switch:
                camera.boost(preset.switch_boost)
            cx, cy, z = camera.update(*obs, conf)
            if i % stride == 0:
                emit(i / meta.fps, (cx, cy, z))
        last_cxyz = (cx, cy, z)

    # always pin the final frame so the renderer interpolates to the very end
    if keyframes and last_index >= 0:
        emit(last_index / meta.fps, (camera.x.value, camera.y.value, camera.zoom.value))

    return CropPath(
        source=meta.source or "",
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        target_aspect=target_aspect,
        preset=preset.name,
        keyframes=keyframes,
    )
