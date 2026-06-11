from reframe.render import sample
from reframe.types import CropKeyframe


def test_sample_interpolates_and_clamps_ends():
    kfs = [CropKeyframe(0.0, 100, 50, 1.0), CropKeyframe(1.0, 200, 150, 2.0)]
    assert sample(kfs, -1.0) == (100, 50, 1.0)  # before first -> first
    assert sample(kfs, 9.0) == (200, 150, 2.0)  # after last -> last
    cx, cy, z = sample(kfs, 0.5)  # midpoint
    assert abs(cx - 150) < 1e-6 and abs(cy - 100) < 1e-6 and abs(z - 1.5) < 1e-6
