"""Core data model. Pure Python (no native deps) so the algorithm is importable
and testable anywhere."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """One detected subject candidate in a single frame (source-pixel coords)."""

    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None
    # optional cues that improve ranking + framing
    mask_area: float | None = None
    has_face: bool = False
    has_pose: bool = False
    face_cx: float | None = None
    face_cy: float | None = None
    motion: float = 0.0  # 0..1 activity (frame-diff inside the box); favors the mover
    # active-speaker cue (lips + audio). Only meaningful when the face is visible
    # enough to mesh; stays 0 otherwise, so it never penalizes — only rewards a talker.
    mouth_open: float | None = None  # per-frame normalized lip aperture (None = no lips meshed)
    lip_motion: float = 0.0  # 0..1 smoothed lip oscillation (the visual "talking" signal)
    speaker_score: float = 0.0  # 0..1 lips moving AND speech present (gated active-speaker)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameDetections:
    """All candidates for one frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int | None = None
    source: str | None = None


@dataclass
class CropKeyframe:
    """A crop window at time `t` (seconds), as a center + zoom in source space.
    A renderer interpolates between keyframes. zoom=1 means the full-height crop
    for the target aspect; zoom>1 is a tighter punch-in."""

    t: float
    cx: float
    cy: float
    zoom: float

    def to_dict(self) -> dict:
        return {"t": round(self.t, 4), "cx": round(self.cx, 2), "cy": round(self.cy, 2), "zoom": round(self.zoom, 4)}


@dataclass
class CropPath:
    """The full reframing result — the contract handed to the renderer."""

    source: str
    fps: float
    width: int
    height: int
    target_aspect: tuple[int, int]
    preset: str
    keyframes: list[CropKeyframe] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "source": self.source,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "target_aspect": list(self.target_aspect),
            "preset": self.preset,
            "keyframes": [k.to_dict() for k in self.keyframes],
        }
