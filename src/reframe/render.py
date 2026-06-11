"""Render a crop path to a final mp4. Optional: the package's job is the crop
path; the renderer is here for standalone use + completeness. Needs the `ml`
extra (opencv) and a system `ffmpeg`.

Pipeline: read frames (cv2) -> interpolate the path -> crop + resize -> pipe raw
frames to ffmpeg (libx264) -> mux the source audio back.
"""

from __future__ import annotations

import bisect
import os
import shutil
import subprocess
import tempfile

from .smooth import base_crop_size, clamp
from .types import CropKeyframe, CropPath


def sample(keyframes: list[CropKeyframe], t: float) -> tuple[float, float, float]:
    """Linearly interpolate (cx, cy, zoom) at time t (seconds)."""
    ts = [k.t for k in keyframes]
    if t <= ts[0]:
        k = keyframes[0]
        return k.cx, k.cy, k.zoom
    if t >= ts[-1]:
        k = keyframes[-1]
        return k.cx, k.cy, k.zoom
    j = bisect.bisect_right(ts, t)
    a, b = keyframes[j - 1], keyframes[j]
    f = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
    return (a.cx + (b.cx - a.cx) * f, a.cy + (b.cy - a.cy) * f, a.zoom + (b.zoom - a.zoom) * f)


def _has_audio(source: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", source],
            capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def render_video(
    path: CropPath,
    source: str,
    out: str,
    out_height: int = 1920,
    out_width: int | None = None,
    crf: int = 20,
    preset: str = "veryfast",
    with_audio: bool = True,
) -> str:  # pragma: no cover (needs cv2 + ffmpeg + a real video)
    """Apply `path` to `source` and write the reframed mp4 to `out`."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError('render_video needs the ml extra: pip install "reframe[ml]"') from e

    aw, ah = path.target_aspect
    if out_width is None:
        out_width = round(out_height * aw / ah)
    out_width -= out_width % 2  # h264 needs even dims
    out_height -= out_height % 2

    base_w, base_h = base_crop_size(path.width, path.height, aw, ah)
    kfs = sorted(path.keyframes, key=lambda k: k.t)

    tmp_video = tempfile.mktemp(suffix=".mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{out_width}x{out_height}", "-pix_fmt", "bgr24", "-r", str(path.fps), "-i", "-",
         "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", tmp_video],
        stdin=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(source)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cx, cy, zoom = sample(kfs, i / path.fps)
            cw = int(clamp(round(base_w / zoom), 16, path.width))
            ch = int(clamp(round(base_h / zoom), 16, path.height))
            x1 = int(clamp(cx - cw / 2, 0, path.width - cw))
            y1 = int(clamp(cy - ch / 2, 0, path.height - ch))
            crop = frame[y1 : y1 + ch, x1 : x1 + cw]
            enc.stdin.write(cv2.resize(crop, (out_width, out_height)).tobytes())
            i += 1
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()

    if with_audio and _has_audio(source):
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_video, "-i", source,
             "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", out],
            check=True, capture_output=True,
        )
        os.remove(tmp_video)
    else:
        shutil.move(tmp_video, out)
    return out
