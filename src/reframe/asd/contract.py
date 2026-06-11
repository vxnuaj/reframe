"""The scores contract — the one thing every ASD backend emits and reframe reads.

Pure-Python, no model deps, so it's importable and testable anywhere. A backend
(any model, any transport) produces a `SpeakingScores`; reframe maps it onto the
subject detections via `score_for_box`. That box-matching is the whole bridge from
"this face is speaking" to "this subject is speaking".

`score` is normalised to 0..1 (1 = confidently speaking). Each backend is
responsible for mapping its model's native output into that range (e.g. a sigmoid
over TalkNet's logits) so the streams are comparable across models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class FaceScore:
    """One face in one frame, with its speaking score (0..1)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    track: int | None = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2), "y1": round(self.y1, 2),
            "x2": round(self.x2, 2), "y2": round(self.y2, 2),
            "score": round(self.score, 4), "track": self.track,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FaceScore:
        return cls(d["x1"], d["y1"], d["x2"], d["y2"], d["score"], d.get("track"))


@dataclass
class FrameScores:
    frame: int
    faces: list[FaceScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"frame": self.frame, "faces": [f.to_dict() for f in self.faces]}

    @classmethod
    def from_dict(cls, d: dict) -> FrameScores:
        return cls(d["frame"], [FaceScore.from_dict(f) for f in d.get("faces", [])])


@dataclass
class SpeakingScores:
    """Per-frame, per-face speaking scores for a whole clip — the contract."""

    source: str
    fps: float
    model: str  # which backend produced this (for provenance / comparison)
    frames: list[FrameScores] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "source": self.source,
            "fps": self.fps,
            "model": self.model,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpeakingScores:
        return cls(
            source=d.get("source", ""),
            fps=d.get("fps", 0.0),
            model=d.get("model", ""),
            frames=[FrameScores.from_dict(f) for f in d.get("frames", [])],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> SpeakingScores:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def by_frame(self) -> dict[int, FrameScores]:
        return {f.frame: f for f in self.frames}

    @staticmethod
    def face_for_box(fs: FrameScores | None, x1: float, y1: float, x2: float, y2: float) -> FaceScore | None:
        """The (highest-scoring) speaking face whose centre sits inside a subject box,
        or None. Used both for the speaker cue and to frame on the stable face box."""
        if fs is None:
            return None
        best: FaceScore | None = None
        for f in fs.faces:
            if x1 <= f.cx <= x2 and y1 <= f.cy <= y2 and (best is None or f.score > best.score):
                best = f
        return best

    @staticmethod
    def score_for_box(fs: FrameScores | None, x1: float, y1: float, x2: float, y2: float) -> float:
        """Speaking score for a subject box: the score of the face inside it (0 if
        none — so a subject with no visible speaking face gets no cue, never a penalty)."""
        f = SpeakingScores.face_for_box(fs, x1, y1, x2, y2)
        return f.score if f is not None else 0.0
