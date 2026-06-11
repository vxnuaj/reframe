"""Shot boundaries → the frame indices where a new scene starts. The camera snaps
(never pans) across these, so getting them right keeps cuts from smearing.

Uses PySceneDetect's AdaptiveDetector when available; otherwise treats the whole
video as one shot (the pipeline still works, just no per-shot reset).
"""

from __future__ import annotations


def scene_starts(video_path: str) -> list[int]:
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError:
        return [0]

    scenes = detect(video_path, AdaptiveDetector())
    starts = [int(start.get_frames()) for start, _end in scenes]
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    return sorted(set(starts))
