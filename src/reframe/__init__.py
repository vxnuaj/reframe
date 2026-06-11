"""reframe — scene-aware intelligent video reframing.

Emits a smooth, subject-tracking *crop path* (data, not pixels) that a renderer
applies. The analysis core (camera, ranker, path) is pure-Python; vision
detectors are an optional `ml` extra.
"""

from .asd import ASDBackend, SpeakingScores, SubprocessASDBackend, apply_speaking_scores, get_backend
from .audio import speech_mask
from .detect import Detector, ReplayDetector
from .path import build_crop_path
from .pipeline import analyze_video, reframe_video
from .presets import DEFAULT_PRESET, PRESETS, Preset, resolve_preset
from .rank import SubjectRanker, SubjectSelector
from .render import render_video, sample
from .scenes import scene_starts
from .smooth import Camera, Follower
from .types import CropKeyframe, CropPath, Detection, FrameDetections, VideoMeta

__version__ = "0.1.0"

__all__ = [
    # core: detections -> crop path
    "build_crop_path",
    "CropPath",
    "CropKeyframe",
    "Detection",
    "FrameDetections",
    "VideoMeta",
    "Preset",
    "PRESETS",
    "DEFAULT_PRESET",
    "resolve_preset",
    "SubjectRanker",
    "SubjectSelector",
    "Camera",
    "Follower",
    "Detector",
    "ReplayDetector",
    "scene_starts",
    "speech_mask",
    # active-speaker backends (hot-swappable; scores contract)
    "SpeakingScores",
    "ASDBackend",
    "SubprocessASDBackend",
    "apply_speaking_scores",
    "get_backend",
    # rendering + one-shot (need the ml extra)
    "render_video",
    "sample",
    "analyze_video",
    "reframe_video",
]
