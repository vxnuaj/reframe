"""Scores contract + merge — pure Python, no model needed."""

from reframe.asd import SpeakingScores, apply_speaking_scores
from reframe.asd.contract import FaceScore, FrameScores
from reframe.detect import ReplayDetector
from reframe.types import Detection, FrameDetections, VideoMeta

W, H, FPS = 1920, 1080, 30.0


def _person(cx, tid):
    return Detection("person", 0.9, cx - 150, 200, cx + 150, 900, track_id=tid)


def test_json_round_trip(tmp_path):
    s = SpeakingScores(
        source="x.mp4", fps=FPS, model="test",
        frames=[FrameScores(0, [FaceScore(100, 100, 200, 200, 0.8, track=1)])],
    )
    p = tmp_path / "scores.json"
    s.save(str(p))
    back = SpeakingScores.load(str(p))
    assert back.model == "test"
    assert back.frames[0].faces[0].score == 0.8
    assert back.frames[0].faces[0].track == 1


def test_score_for_box_matches_by_centre():
    fs = FrameScores(0, [
        FaceScore(620, 250, 780, 410, 0.9),   # centre 700,330 -> inside left person
        FaceScore(1120, 250, 1280, 410, 0.1),  # centre 1200,330 -> inside right person
    ])
    # left person box around cx=700, right around cx=1200
    assert SpeakingScores.score_for_box(fs, 550, 200, 850, 900) == 0.9
    assert SpeakingScores.score_for_box(fs, 1050, 200, 1350, 900) == 0.1
    # no face in this box -> 0, never a penalty
    assert SpeakingScores.score_for_box(fs, 0, 0, 100, 100) == 0.0
    assert SpeakingScores.score_for_box(None, 0, 0, 100, 100) == 0.0


def test_apply_sets_speaker_score_on_subjects():
    # two people; the ASD scores say the left one (cx=700) is speaking
    frames = [FrameDetections(0, [_person(700, tid=1), _person(1200, tid=2)])]
    meta = VideoMeta(fps=FPS, width=W, height=H, source="x")
    det = ReplayDetector(meta, frames, track=False)
    scores = SpeakingScores(
        source="x", fps=FPS, model="test",
        frames=[FrameScores(0, [FaceScore(640, 250, 760, 370, 0.85, track=1)])],
    )
    out = list(apply_speaking_scores(det.frames(), scores))
    left = next(d for d in out[0].detections if d.track_id == 1)
    right = next(d for d in out[0].detections if d.track_id == 2)
    assert left.speaker_score == 0.85
    assert right.speaker_score == 0.0
