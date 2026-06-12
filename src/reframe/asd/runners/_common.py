"""Shared ASD-runner pipeline (TalkNet family: LR-ASD, TalkNet, fast-asd).

These models share lineage — S3FD faces, IOU tracking, 224 face-crop, MFCC audio,
TalkNet-style windowed scoring at 25 fps. This module holds all of that; a runner
only supplies the model-specific `load_model` + `score_track` and calls `run()`.

Device-agnostic (CUDA > MPS > CPU). Emits the reframe SpeakingScores contract in
the SOURCE video's native frame numbering (mapping the 25fps working rate back).
Must run with the model's repo on PYTHONPATH and cwd = repo root (so weights and
`model.faceDetector.s3fd` resolve).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import tempfile

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

PROC_FPS = 25  # TalkNet-family internal working frame rate


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_state(sd: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _bb_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _track_shot(faces, min_track=10, num_failed=10, min_face=1, iou_thres=0.5):
    faces = [list(f) for f in faces]
    tracks = []
    while True:
        track = []
        for frame_faces in faces:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= num_failed:
                    if _bb_iou(face["bbox"], track[-1]["bbox"]) > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > min_track:
            fn = np.array([f["frame"] for f in track])
            bb = np.array([np.array(f["bbox"]) for f in track])
            fi = np.arange(fn[0], fn[-1] + 1)
            bbi = np.stack([interp1d(fn, bb[:, j])(fi) for j in range(4)], axis=1)
            if max(np.mean(bbi[:, 2] - bbi[:, 0]), np.mean(bbi[:, 3] - bbi[:, 1])) > min_face:
                tracks.append({"frame": fi, "bbox": bbi})
    return tracks


def _log(msg):
    print(f"[reframe-asd] {msg}", flush=True)


def _detect_faces(det, frames_dir, scale=0.25):
    flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    dets = []
    n = len(flist)
    _log(f"face detection (S3FD): {n} frames")
    for fidx, fname in enumerate(flist):
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        bboxes = det.detect_faces(img, conf_th=0.9, scales=[scale])
        dets.append([{"frame": fidx, "bbox": b[:-1].tolist(), "conf": b[-1]} for b in bboxes])
        if n and (fidx + 1) % 100 == 0:
            _log(f"face detection: {fidx + 1}/{n} frames")
    return dets, flist


def _track_features(track, flist, audio, crop_scale=0.40):
    s = signal.medfilt([max(d[3] - d[1], d[2] - d[0]) / 2 for d in track["bbox"]], 13)
    y = signal.medfilt([(d[1] + d[3]) / 2 for d in track["bbox"]], 13)
    x = signal.medfilt([(d[0] + d[2]) / 2 for d in track["bbox"]], 13)
    vid = []
    for fidx, frame in enumerate(track["frame"]):
        bs = s[fidx]
        bsi = int(bs * (1 + 2 * crop_scale))
        image = np.pad(cv2.imread(flist[int(frame)]), ((bsi, bsi), (bsi, bsi), (0, 0)),
                       "constant", constant_values=(110, 110))
        my, mx = y[fidx] + bsi, x[fidx] + bsi
        face = image[int(my - bs):int(my + bs * (1 + 2 * crop_scale)),
                     int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
        face = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
        vid.append(face[56:168, 56:168])
    a0 = int(track["frame"][0] / PROC_FPS * 16000)
    a1 = int((track["frame"][-1] + 1) / PROC_FPS * 16000)
    mfcc = python_speech_features.mfcc(audio[a0:a1], 16000, numcep=13, winlen=0.025, winstep=0.010)
    return np.array(vid), mfcc


def run(load_model, score_track, default_weight):
    """Drive the pipeline. `load_model(weight, dev)->ctx`; `score_track(ctx, vfeat,
    afeat, dev)->per-frame raw score array` (logit; we sigmoid + smooth here)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="asd")
    ap.add_argument("--weight", default=default_weight)
    ap.add_argument("--s3fd-weight", default=None, help="load the S3FD face weights from here")
    args = ap.parse_args()

    if args.s3fd_weight:
        import model.faceDetector.s3fd as _s3fd

        _s3fd.PATH_WEIGHT = args.s3fd_weight
    from model.faceDetector.s3fd import S3FD

    import time

    t0 = time.time()
    dev = pick_device()
    native_fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 25.0
    _log(f"start | model={args.model} | device={dev} | loading weights")
    ctx = load_model(args.weight, dev)
    det = S3FD(device=dev)

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir)
        audio_path = os.path.join(tmp, "audio.wav")
        _log(f"extracting frames @ {PROC_FPS}fps + audio (ffmpeg)")
        subprocess.run(["ffmpeg", "-y", "-i", args.video, "-qscale:v", "2", "-r", str(PROC_FPS),
                        "-f", "image2", os.path.join(frames_dir, "%06d.jpg"), "-loglevel", "error"], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", args.video, "-ac", "1", "-vn", "-ar", "16000",
                        audio_path, "-loglevel", "error"], check=True)
        _, audio = wavfile.read(audio_path)

        dets, flist = _detect_faces(det, frames_dir)
        tracks = _track_shot(dets)
        _log(f"tracking: {len(tracks)} face track(s) across {len(flist)} frames | scoring (TalkNet)")

        frames: dict[int, list] = {}
        for tidx, track in enumerate(tracks):
            _log(f"scoring track {tidx + 1}/{len(tracks)} ({len(track['frame'])} frames)")
            vfeat, afeat = _track_features(track, flist, audio)
            scores = score_track(ctx, vfeat, afeat, dev)
            for fidx, frame25 in enumerate(track["frame"].tolist()):
                if fidx >= len(scores):
                    break
                window = scores[max(fidx - 2, 0): min(fidx + 3, len(scores))]
                prob = 1.0 / (1.0 + math.exp(-float(np.mean(window))))
                x1, y1, x2, y2 = (float(v) for v in track["bbox"][fidx])
                nframe = int(round(frame25 / PROC_FPS * native_fps))
                frames.setdefault(nframe, []).append(
                    {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                     "score": round(prob, 4), "track": tidx})

    out = {"version": 1, "source": os.path.basename(args.video), "fps": native_fps,
           "model": args.model, "frames": [{"frame": k, "faces": v} for k, v in sorted(frames.items())]}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    _log(f"done | {len(tracks)} tracks, {len(frames)} scored frames | {time.time() - t0:.1f}s | device={dev}")
