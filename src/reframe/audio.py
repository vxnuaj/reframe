"""Optional audio cue for active-speaker detection: a per-frame speech mask.

This does NOT say *who* is speaking (the lips do that) — only *when* there's
speech, so the lip-motion cue can be gated to actual talking instead of crediting
chewing/laughing/nodding during silence. It's a plain energy VAD over the
extracted mono track: no dependency beyond ffmpeg + numpy. Degrades to None
("no usable audio") when there's no track or extraction fails, and callers then
simply skip audio gating.

Energy VAD triggers on any loud sound (music, claps), not just voice — fine for
talk-style footage where the point is "is anyone speaking right now". Swap in a
real VAD (webrtcvad / silero) later if footage needs it.
"""

from __future__ import annotations

import subprocess


def speech_mask(video_path: str, fps: float) -> list[bool] | None:
    """Per-frame speech presence, indexed by video frame number. Returns None if
    there's no usable audio (caller should then not gate on audio at all)."""
    import numpy as np

    sr = 16000
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", video_path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    if not proc.stdout:
        return None

    audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        return None

    hop = sr / max(fps, 1.0)
    n = int(audio.size / hop) + 1
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = audio[int(i * hop) : int((i + 1) * hop)]
        if seg.size:
            rms[i] = float(np.sqrt(np.mean(seg * seg)))

    # adaptive threshold between the noise floor and speech peaks
    floor = float(np.percentile(rms, 20))
    peak = float(np.percentile(rms, 95))
    if peak <= floor:
        return None
    thresh = floor + 0.15 * (peak - floor)
    mask = (rms > thresh).tolist()
    # hold speech across short inter-word gaps so it doesn't flicker off mid-sentence
    return _hold(mask, int(round(fps * 0.25)))


def _hold(mask: list[bool], hold: int) -> list[bool]:
    out = list(mask)
    last = -(10**9)
    for i, v in enumerate(mask):
        if v:
            last = i
        elif i - last <= hold:
            out[i] = True
    return out
