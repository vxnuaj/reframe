"""A minimal greedy-IoU tracker. Used to assign stable track IDs when the
detector doesn't supply them (e.g. plain per-frame detection). When the detector
already tracks (YOLO + ByteTrack), skip this and use its IDs.

Greedy IoU is intentionally simple; it's enough for the single-/few-subject
short-form case. Swap in Hungarian/ByteTrack later if subjects cross often.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Detection

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    id: int
    box: Box
    age: int = 0  # frames since last matched


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        """Assign det.track_id in place, return the same list."""
        matched: set[int] = set()
        # sort by area desc so bigger subjects claim tracks first
        for d in sorted(dets, key=lambda x: x.area, reverse=True):
            box: Box = (d.x1, d.y1, d.x2, d.y2)
            best, best_iou = None, self.iou_threshold
            for t in self.tracks:
                if t.id in matched:
                    continue
                v = iou(box, t.box)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is None:
                best = _Track(self._next_id, box)
                self._next_id += 1
                self.tracks.append(best)
            best.box = box
            best.age = 0
            matched.add(best.id)
            d.track_id = best.id
        for t in self.tracks:
            if t.id not in matched:
                t.age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return dets
