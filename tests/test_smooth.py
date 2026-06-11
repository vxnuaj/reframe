from reframe.smooth import Camera, Follower, base_crop_size
from reframe.presets import PRESETS


def test_follower_respects_step_cap_and_converges():
    f = Follower(value=0.0, response=0.2, damping=0.8, target_alpha=0.5)
    prev = f.value
    for _ in range(400):
        v = f.update(observation=100.0, confidence=1.0, max_step=5.0, lo=-1e4, hi=1e4)
        assert abs(v - prev) <= 5.0 + 1e-6  # never exceeds the per-frame step cap
        prev = v
    assert abs(f.value - 100.0) < 1.0  # eventually reaches the target


def test_low_confidence_moves_more_gently():
    hi = Follower(0.0, 0.3, 0.8, 0.3)
    lo = Follower(0.0, 0.3, 0.8, 0.3)
    hi_step = abs(hi.update(100.0, confidence=1.0, max_step=8.0, lo=-1e4, hi=1e4))
    lo_step = abs(lo.update(100.0, confidence=0.0, max_step=8.0, lo=-1e4, hi=1e4))
    assert lo_step < hi_step  # uncertain detection => smaller nudge


def test_snap_clears_velocity():
    f = Follower(0.0, 0.3, 0.8, 0.3)
    for _ in range(10):
        f.update(100.0, 1.0, 8.0, -1e4, 1e4)
    assert f.velocity != 0.0
    f.snap(500.0)
    assert f.value == 500.0 and f.velocity == 0.0 and f.target == 500.0


def test_base_crop_16x9_to_9x16_is_full_height():
    w, h = base_crop_size(1920, 1080, 9, 16)
    assert h == 1080
    assert abs(w / h - 9 / 16) < 0.01


def test_camera_keeps_crop_inside_frame():
    cam = Camera(1920, 1080, (9, 16), PRESETS["talking_head"], 960, 540, 1.1)
    for obs in (0.0, 5000.0, 960.0):  # off-frame observations
        cx, cy, z = cam.update(obs, 540, 1.1, 1.0)
        cw, ch = cam.crop_size(z)
        assert cw / 2 <= cx <= 1920 - cw / 2
        assert ch / 2 <= cy <= 1080 - ch / 2
