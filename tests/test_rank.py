"""Active-speaker cue behaviour in the ranker — pure Python, no ML needed."""

from reframe.rank import SubjectRanker, SubjectSelector
from reframe.types import Detection

W, H = 1920, 1080


def _at(cx, speaker_score=0.0, motion=0.0, tid=1):
    return Detection(
        "person", 0.9, cx - 150, 200, cx + 150, 900,
        track_id=tid, mask_area=300 * 700, speaker_score=speaker_score, motion=motion,
    )


def _person(cx, speaker_score=0.0, motion=0.0, tid=1):
    return Detection(
        "person", 0.9, cx - 150, 200, cx + 150, 900,
        track_id=tid, mask_area=300 * 700, speaker_score=speaker_score, motion=motion,
    )


def test_talker_outranks_silent_peer():
    ranker = SubjectRanker()
    # two equally-framed people; one is the active speaker (visible moving lips)
    talker = _person(700, speaker_score=0.8, tid=1)
    listener = _person(1200, speaker_score=0.0, tid=2)
    assert ranker.score(talker, W, H, None) > ranker.score(listener, W, H, None)


def test_speaker_cue_is_off_when_zero():
    # speaker_score=0 must add nothing — two identical people tie regardless
    ranker = SubjectRanker()
    a = _person(900, speaker_score=0.0, tid=1)
    b = _person(900, speaker_score=0.0, tid=2)
    assert ranker.score(a, W, H, None) == ranker.score(b, W, H, None)


def test_lips_beat_a_fidgeting_listener():
    # the talker is still-bodied but speaking; the listener fidgets (motion) but
    # is silent. The speaker cue (1.6) should outweigh the motion cue (1.4).
    ranker = SubjectRanker()
    talker = _person(700, speaker_score=0.8, motion=0.0, tid=1)
    fidgeter = _person(700, speaker_score=0.0, motion=0.8, tid=2)
    assert ranker.score(talker, W, H, None) > ranker.score(fidgeter, W, H, None)


def test_selector_follows_a_clear_speaker_turn():
    # A is dead-centre (wins the initial lock); then B starts clearly talking.
    # The lock should move to B once the smoothed speaker signal is established.
    sel = SubjectSelector()
    a_central, b_side = 960, 1300
    for _ in range(15):  # warm-up + hold: acquires the central, silent A
        sel.select([_at(a_central, tid=1), _at(b_side, tid=2)], W, H)
    assert sel.locked_track_id == 1
    for _ in range(40):  # B takes the turn and talks
        sel.select([_at(a_central, tid=1), _at(b_side, speaker_score=1.0, tid=2)], W, H)
    assert sel.locked_track_id == 2  # followed the speaker


def test_selector_ignores_a_silent_fidget():
    # B fidgets (motion) but never speaks; it must not steal the lock from A.
    sel = SubjectSelector()
    a_central, b_side = 960, 1300
    for _ in range(15):
        sel.select([_at(a_central, tid=1), _at(b_side, motion=0.5, tid=2)], W, H)
    first = sel.locked_track_id
    for _ in range(40):
        sel.select([_at(a_central, tid=1), _at(b_side, motion=0.5, tid=2)], W, H)
    assert sel.locked_track_id == first
