import math

from reframe.detect import ReplayDetector
from reframe.path import build_crop_path
from reframe.presets import PRESETS
from reframe.types import Detection, FrameDetections, VideoMeta

W, H, FPS = 1920, 1080, 30.0
MAX_STEP_X = PRESETS["talking_head"].max_step_x  # per-frame pan cap (pre confidence scaling)


def _person(cx, cy=540, w=300, h=700, conf=0.9, tid=1):
    return Detection("person", conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, track_id=tid)


def _path_for(frames, scene_starts=(0,)):
    meta = VideoMeta(fps=FPS, width=W, height=H, frame_count=len(frames), source="syn")
    det = ReplayDetector(meta, frames, track=False)
    return build_crop_path(
        meta, det.frames(), scene_starts=list(scene_starts),
        preset_name="talking_head", target_aspect=(9, 16), keyframe_fps=FPS,
    )


def test_follows_moving_subject_within_step_cap():
    n = 120
    frames = []
    for i in range(n):
        sx = 960 + 460 * math.sin(i / n * math.pi)  # pans right then back
        frames.append(FrameDetections(i, [_person(sx)]))
    path = _path_for(frames)
    kfs = path.keyframes
    assert len(kfs) >= n - 1

    # 1) never whips: per-frame center step bounded by the preset's max_step_x
    for a, b in zip(kfs, kfs[1:]):
        assert abs(b.cx - a.cx) <= MAX_STEP_X + 1e-3

    # 2) actually follows: camera has panned right by mid-clip
    assert kfs[len(kfs) // 2].cx > kfs[0].cx + 50

    # 3) crop stays in-frame
    for k in kfs:
        assert 0 < k.cx < W and 0 < k.cy < H and k.zoom >= 1.0


def test_scene_cut_snaps_camera():
    # scene 0: subject parked left; scene 1 (frame 30): subject jumps right
    frames = []
    for i in range(30):
        frames.append(FrameDetections(i, [_person(400, tid=1)]))
    for i in range(30, 60):
        frames.append(FrameDetections(i, [_person(1500, tid=2)]))
    path = _path_for(frames, scene_starts=(0, 30))
    by_frame = {round(k.t * FPS): k for k in path.keyframes}

    pre, post = by_frame[29], by_frame[30]
    # the cut snaps: a jump far larger than the per-frame step cap
    assert post.cx - pre.cx > 100
    # and within each shot it's smooth again
    assert abs(by_frame[31].cx - by_frame[30].cx) <= MAX_STEP_X + 1e-3


def test_graceful_hold_when_subject_missing():
    # subject for 20 frames, then nothing for 20 — camera should hold, not jump
    frames = [FrameDetections(i, [_person(1300)]) for i in range(20)]
    frames += [FrameDetections(i, []) for i in range(20, 40)]
    path = _path_for(frames)
    by_frame = {round(k.t * FPS): k for k in path.keyframes}
    held_start = by_frame[20].cx
    held_end = by_frame[39].cx
    assert abs(held_end - held_start) < 30  # barely drifts while subject is gone
