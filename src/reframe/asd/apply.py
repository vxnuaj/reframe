"""Merge a SpeakingScores stream onto a detection stream.

This is the seam where ASD output becomes a ranker cue: for each frame, set every
subject's `speaker_score` from the speaking face that sits inside its box. Pure
Python — the detector and the model never meet, they only meet here, by frame index.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..types import FrameDetections
from .contract import SpeakingScores


def apply_speaking_scores(
    frames: Iterable[FrameDetections], scores: SpeakingScores
) -> Iterator[FrameDetections]:
    by_frame = scores.by_frame()
    for fd in frames:
        fs = by_frame.get(fd.frame_index)
        for d in fd.detections:
            face = SpeakingScores.face_for_box(fs, d.x1, d.y1, d.x2, d.y2)
            if face is not None:
                d.speaker_score = face.score
                # frame on the ASD face box (medfilt-smoothed in the runner) instead
                # of a jittery per-frame face detector — stable head framing.
                d.has_face = True
                d.face_cx = face.cx
                d.face_cy = face.cy
            else:
                d.speaker_score = 0.0
        yield fd
