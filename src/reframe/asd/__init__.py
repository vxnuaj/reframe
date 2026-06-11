"""Active-speaker detection backends.

The reframe core never imports a model. It only speaks the *scores contract* — a
normalized per-frame, per-face speaking score (see `contract.py`). Each ASD model
(TalkNet, fast-asd, LR-ASD, ...) lives in its own repo/environment and is reached
through a backend (see `base.py`) that emits that contract — a subprocess into the
model's own venv now, an HTTP service in production. Swapping models never touches
the ranker, camera, or crop-path.
"""

from __future__ import annotations

from .apply import apply_speaking_scores
from .backends import BACKENDS, get_backend
from .base import ASDBackend, SubprocessASDBackend
from .contract import FaceScore, FrameScores, SpeakingScores

__all__ = [
    "SpeakingScores",
    "FrameScores",
    "FaceScore",
    "ASDBackend",
    "SubprocessASDBackend",
    "apply_speaking_scores",
    "get_backend",
    "BACKENDS",
]
