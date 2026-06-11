"""Pick *which* subject to follow, frame to frame, with continuity.

Two pieces:
  - SubjectRanker: a cheap, transparent linear scorer over per-detection features
    (confidence, size, centeredness, face/pose/speaker cues, and whether it's the
    currently-locked track). No model, no LLM — just weighted features.
  - SubjectSelector: applies hysteresis so the camera doesn't flicker between
    people: it locks onto a subject and only switches when a clearly better one
    appears AND a minimum hold has elapsed; it tolerates short dropouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .smooth import clamp
from .types import Detection


def _default_class_bias() -> dict[str, float]:
    return {
        "person": 0.22,
        "dog": 0.12,
        "cat": 0.10,
        "car": 0.06,
        "bicycle": 0.02,
        "motorcycle": 0.02,
        "bus": 0.01,
        "truck": 0.01,
        "sports ball": 0.05,
    }


@dataclass
class RankWeights:
    det_conf: float = 1.35
    area: float = 0.95
    center_affinity: float = 0.55
    face: float = 0.48
    pose: float = 0.34
    speaker: float = 1.6  # active speaker (visible moving lips + speech) — a strong cue
    motion: float = 1.0  # the mover (writing/gesturing); kept below speaker so a
    #                      reacting listener can't out-vote the actual talker
    size_logit: float = 0.26
    lock_match: float = 1.3
    class_bias: dict[str, float] = field(default_factory=_default_class_bias)


class SubjectRanker:
    def __init__(self, weights: RankWeights | None = None) -> None:
        self.w = weights or RankWeights()

    def score(self, d: Detection, frame_w: int, frame_h: int, locked_track_id: int | None) -> float:
        w = self.w
        frame_area = max(1.0, frame_w * frame_h)
        norm_area = clamp(d.area / frame_area, 0.0, 1.0)
        diag = math.hypot(frame_w, frame_h)
        dist_center = math.hypot(d.cx - frame_w / 2, d.cy - frame_h / 2)
        center_affinity = 1.0 - clamp(dist_center / max(diag, 1.0), 0.0, 1.0)

        s = w.class_bias.get(d.cls_name, 0.0)
        s += w.det_conf * clamp(d.conf, 0.0, 1.0)
        s += w.area * math.sqrt(norm_area)
        s += w.center_affinity * center_affinity
        s += w.face * (1.0 if d.has_face else 0.0)
        s += w.pose * (1.0 if d.has_pose else 0.0)
        s += w.speaker * clamp(d.speaker_score, 0.0, 1.0)
        s += w.motion * clamp(d.motion, 0.0, 1.0)
        s += w.size_logit * math.log1p(norm_area * 250.0)
        if d.track_id is not None and d.track_id == locked_track_id:
            s += w.lock_match
        return s


class SubjectSelector:
    def __init__(
        self,
        ranker: SubjectRanker | None = None,
        min_hold_frames: int = 18,
        max_missed_frames: int = 40,
        switch_margin: float = 0.8,
        warmup_frames: int = 12,
        motion_alpha: float = 0.12,
        speaker_alpha: float = 0.08,
        speaker_floor: float = 0.15,
        speaker_switch_margin: float = 0.1,
        turn_hold_frames: int = 48,
        reassoc_radius: float = 0.15,
    ) -> None:
        self.ranker = ranker or SubjectRanker()
        self.min_hold = min_hold_frames
        self.max_missed = max_missed_frames
        # switch_margin + min_hold are the anti-flicker hysteresis: a challenger must
        # beat the locked subject by this much AND the hold must have elapsed. Tuned
        # up for two-person talk footage, where the speaker signal swings on every
        # word — without firm hysteresis the lock ping-pongs between the two heads.
        self.switch_margin = switch_margin
        # warmup_frames: at a scene start, re-acquire freely (no hysteresis) for
        # this many frames so the activity signal has time to accumulate before we
        # commit a lock — this is what keeps us from locking a still bystander on
        # frame one (where motion is 0 for everyone).
        self.warmup_frames = warmup_frames
        # motion_alpha: EMA weight for per-track activity. The raw per-frame motion
        # cue is spiky (a speaker writes, then pauses); smoothing it over a window
        # (~1/alpha frames) turns intermittent gestures into a *persistent* signal,
        # so the mover's advantage holds every frame instead of clearing the switch
        # margin only by luck. This is what kills the multi-second lock-on delay.
        self.motion_alpha = motion_alpha
        # speaker_alpha: a SLOWER EMA (~0.5s) over the active-speaker score. The raw
        # lip signal drops to zero between syllables; without this the "current
        # talker" flickers off mid-sentence and a fidgety listener grabs the lock.
        # Slow smoothing keeps a talker "on" across the whole turn.
        self.speaker_alpha = speaker_alpha
        # speaker-driven switching: follow a *clearly* speaking challenger (>= floor)
        # that out-talks the locked subject by speaker_switch_margin, on a much
        # smaller bar than the general switch_margin. The big margin stops a fidgety
        # listener from stealing focus; this small one lets the camera actually
        # follow turn-taking, since the smoothed speaker signal only crosses on a
        # real change of speaker (not per-syllable).
        self.speaker_floor = speaker_floor
        self.speaker_switch_margin = speaker_switch_margin
        # turn_hold_frames: a challenger must be the CLEAR dominant speaker for this
        # many (net) frames before it takes the lock. This is the "hold the floor"
        # debounce — it's what stops the camera from chasing every backchannel and
        # ping-ponging between two heads. Brief blips build evidence that decays;
        # only a sustained turn change accumulates enough to switch.
        self.turn_hold = turn_hold_frames
        # reassoc_radius: fraction of frame width. When the tracker drops/reuses our
        # subject's id, a detection within this radius of where the subject just was
        # is treated as the SAME subject (bridges id churn instead of re-picking).
        self.reassoc_radius = reassoc_radius
        self.reset()

    def reset(self) -> None:
        """Clear continuity state — call at a scene boundary so a new shot
        re-acquires its own subject."""
        self.locked_track_id: int | None = None
        self.frames_on_subject = 0
        self.missed = 0
        self.frames_since_reset = 0
        self._activity: dict[int, float] = {}
        self._speaking: dict[int, float] = {}
        self._chal_tid: int | None = None  # who's currently trying to take the floor
        self._chal_streak = 0  # accumulated evidence they've taken it
        self._last_pos: tuple[float, float] | None = None  # last subject centre (id-churn bridge)
        self.last_switch = False  # did THIS select() commit a real subject change?

    def _smooth_motion(self, dets: list[Detection]) -> None:
        """Replace each tracked detection's instantaneous motion with a per-track
        EMA, so the ranker sees sustained activity rather than single-frame spikes.
        Untracked detections keep their raw motion."""
        a = self.motion_alpha
        for d in dets:
            if d.track_id is None:
                continue
            prev = self._activity.get(d.track_id, d.motion)
            ema = prev * (1.0 - a) + d.motion * a
            self._activity[d.track_id] = ema
            d.motion = ema

    def _smooth_speaker(self, dets: list[Detection]) -> None:
        """Replace the per-frame speaker score with a slow per-track EMA, so the
        'current talker' is stable across a turn rather than blinking on each word."""
        a = self.speaker_alpha
        for d in dets:
            if d.track_id is None:
                continue
            prev = self._speaking.get(d.track_id, d.speaker_score)
            ema = prev * (1.0 - a) + d.speaker_score * a
            self._speaking[d.track_id] = ema
            d.speaker_score = ema

    def select(self, dets: list[Detection], frame_w: int, frame_h: int) -> Detection | None:
        if not dets:
            self.missed += 1
            if self.missed > self.max_missed:
                self.locked_track_id = None
                self.frames_on_subject = 0
                self._last_pos = None
            return None  # caller holds the last camera position (graceful fallback)
        self.frames_since_reset += 1
        self.last_switch = False  # set True only on a real turn change below
        self._smooth_motion(dets)
        self._smooth_speaker(dets)

        # Rank on RAW scores (no lock bonus) so a clearly-better subject can win.
        # The lock's stickiness comes from the hysteresis below — NOT from blinding
        # the selector to challengers.
        def raw(d: Detection) -> float:
            return self.ranker.score(d, frame_w, frame_h, None)

        best = max(dets, key=raw)
        locked = next((d for d in dets if d.track_id is not None and d.track_id == self.locked_track_id), None)

        # Continuity bridge: the tracker dropped or reused our subject's id, but a box
        # is still sitting where the subject just was — adopt it as the same person.
        # Without this, a one-frame id churn drops us into re-acquisition and the lock
        # jumps to the other head (a big source of the "flicker").
        if locked is None and self.locked_track_id is not None and self._last_pos is not None:
            lx, ly = self._last_pos
            cand = min(dets, key=lambda d: math.hypot(d.cx - lx, d.cy - ly))
            if math.hypot(cand.cx - lx, cand.cy - ly) <= self.reassoc_radius * frame_w:
                self.locked_track_id = cand.track_id  # same subject, new id; keep the hold
                locked = cand

        # Acquire: scene start / warm-up / never-locked. Follow the running best; as
        # the speaker signal builds, best migrates onto the talker.
        if self.locked_track_id is None or self.frames_since_reset <= self.warmup_frames:
            self.missed = 0
            self._lock(best)
            self._last_pos = (best.cx, best.cy)
            return best

        # We hold a lock id but the subject isn't visible and didn't re-associate.
        # Hold the last framing for a few frames rather than jumping to the other
        # person on a brief tracker dropout; only re-acquire if it's gone for good.
        if locked is None:
            self.missed += 1
            if self.missed <= self.max_missed:
                return None
            self.missed = 0
            self._lock(best)
            self._last_pos = (best.cx, best.cy)
            return best
        self.missed = 0

        subject = locked if best is locked else self._resolve_switch(dets, locked, best, raw)
        if best is locked:
            self.frames_on_subject += 1
        self._last_pos = (subject.cx, subject.cy)
        return subject

    def _resolve_switch(self, dets, locked, best, raw):
        # Hold-the-floor switch: a challenger takes the lock only after being the
        # CLEAR dominant speaker for a sustained stretch (turn_hold), not the instant
        # its score edges ahead. Evidence accumulates +1 while one challenger clearly
        # out-talks the locked subject, decaying otherwise. Brief interjections never
        # accumulate enough; a real turn change does. Ambiguous (both ~equal — common
        # at low res) -> nothing happens, we stay.
        top = max(dets, key=lambda d: d.speaker_score)
        dominant = (
            top is not locked
            and top.track_id is not None
            and top.speaker_score >= self.speaker_floor
            and top.speaker_score > locked.speaker_score + self.speaker_switch_margin
        )
        if dominant and top.track_id == self._chal_tid:
            self._chal_streak += 1
        elif dominant:
            self._chal_tid = top.track_id
            self._chal_streak = 1
        else:
            self._chal_streak -= 1
            if self._chal_streak <= 0:
                self._chal_streak = 0
                self._chal_tid = None
        if self._chal_tid is not None and self._chal_streak >= self.turn_hold:
            new = next((d for d in dets if d.track_id == self._chal_tid), None)
            if new is not None:
                self._lock(new)
                self._chal_tid = None
                self._chal_streak = 0
                self.last_switch = True  # real turn change (a cut)
                return new
        # Non-speaker challenger (motion/size/centre): switch only on the LARGE margin
        # after the hold. Guards the lock from a mover that isn't the speaker.
        if self.frames_on_subject >= self.min_hold and raw(best) > raw(locked) + self.switch_margin:
            self._lock(best)
            self.last_switch = True  # real subject change (a cut)
            return best
        self.frames_on_subject += 1
        return locked

    def _lock(self, det: Detection) -> None:
        if det.track_id != self.locked_track_id:
            self.locked_track_id = det.track_id
            self.frames_on_subject = 1
        else:
            self.frames_on_subject += 1
