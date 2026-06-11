"""Detectors produce a per-frame stream of candidate subjects. The reframing
core doesn't care *how* — only that something yields FrameDetections + VideoMeta.
So the backend is swappable: bundled YOLO here, or a cloud vision service, or
canned data for tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .track import IoUTracker
from .types import Detection, FrameDetections, VideoMeta


class Detector(Protocol):
    @property
    def meta(self) -> VideoMeta: ...

    def frames(self) -> Iterator[FrameDetections]: ...


class ReplayDetector:
    """Feeds pre-computed detections. Pure Python — used by tests and by anyone
    who already has boxes (e.g. from a cloud vision API)."""

    def __init__(self, meta: VideoMeta, frames: list[FrameDetections], track: bool = True) -> None:
        self._meta = meta
        self._frames = frames
        self._tracker = IoUTracker() if track else None

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def frames(self) -> Iterator[FrameDetections]:
        for fd in self._frames:
            if self._tracker is not None and any(d.track_id is None for d in fd.detections):
                self._tracker.update(fd.detections)
            yield fd


class YoloDetector:
    """Bundled local backend: YOLO (Ultralytics) + ByteTrack for tracked person/
    object boxes, with optional MediaPipe face refinement for better head framing.

    Requires the `ml` extra:  pip install "reframe[ml]"
    """

    def __init__(
        self,
        video_path: str,
        classes: tuple[str, ...] = ("person",),
        model: str = "yolo11n.pt",
        conf: float = 0.3,
        use_face: bool = True,
        use_lips: bool = True,
    ) -> None:
        try:
            import cv2  # noqa: F401
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError('YoloDetector needs the ml extra: pip install "reframe[ml]"') from e

        import cv2

        self.video_path = video_path
        self.classes = classes
        self.conf = conf
        self.use_face = use_face
        self.use_lips = use_lips
        self._model = YOLO(model)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
        self._meta = VideoMeta(fps=fps, width=w, height=h, frame_count=n, source=video_path)

        self._face = None
        self._mesh = None
        self._speech: list[bool] | None = None
        if use_face:
            # Optional face cue: full-range BlazeFace (model_selection=1) via
            # mediapipe's legacy solutions API (the reason ml pins 0.10.14). Full
            # range catches mid-distance faces (e.g. podcast subjects); short range
            # only catches close-ups. Degrades gracefully if unavailable.
            try:
                import mediapipe as mp

                self._face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            except Exception as e:
                print(f"[reframe] face refinement disabled ({type(e).__name__}: {e})")
                self._face = None

            # Active-speaker cue: Face Mesh gives detailed lip landmarks so we can
            # measure mouth movement. Its short-range detector is also the *gate* —
            # if a face is too small/turned to mesh, there's simply no lip signal,
            # which is exactly "we can't see the lips moving" -> cue stays off.
            # Skipped entirely when an external ASD backend supplies speaker_score.
            if use_lips:
                try:
                    import mediapipe as mp

                    self._mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=5,
                        refine_landmarks=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                except Exception as e:
                    print(f"[reframe] lip / speaker cue disabled ({type(e).__name__}: {e})")
                    self._mesh = None

            # Audio gate: per-frame speech mask so lip motion only counts as talking
            # when there's actual speech (None = no usable audio -> don't gate).
            if self._mesh is not None:
                from .audio import speech_mask

                self._speech = speech_mask(video_path, self._meta.fps)

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def frames(self) -> Iterator[FrameDetections]:  # pragma: no cover (needs ml + video)
        import cv2

        name_to_id = {v: k for k, v in self._model.names.items()}
        keep_ids = [name_to_id[c] for c in self.classes if c in name_to_id]

        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        prev_gray = None
        prev_open: dict[int, float] = {}  # last lip aperture per track
        lip_ema: dict[int, float] = {}  # smoothed lip motion per track
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = self._model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=keep_ids or None, conf=self.conf, verbose=False
            )
            dets: list[Detection] = []
            r = results[0]
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    cls_id = int(b.cls[0])
                    tid = int(b.id[0]) if b.id is not None else None
                    dets.append(
                        Detection(
                            cls_name=self._model.names.get(cls_id, str(cls_id)),
                            conf=float(b.conf[0]),
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            track_id=tid,
                            mask_area=(x2 - x1) * (y2 - y1),
                        )
                    )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and dets:
                self._set_motion(gray, prev_gray, dets)
            prev_gray = gray
            if self._face is not None and dets:
                self._refine_faces(frame, dets)
            if self._mesh is not None and dets:
                speech = self._speech[idx] if (self._speech is not None and idx < len(self._speech)) else True
                self._set_speaker(frame, dets, speech, prev_open, lip_ema)
            yield FrameDetections(frame_index=idx, detections=dets)
            idx += 1
        cap.release()

    def _set_motion(self, gray, prev_gray, dets: list[Detection]) -> None:  # pragma: no cover
        import cv2

        h, w = gray.shape[:2]
        for d in dets:
            x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
            x2, y2 = min(w, int(d.x2)), min(h, int(d.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            diff = cv2.absdiff(gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2])
            d.motion = min(float(diff.mean()) / 10.0, 1.0)

    # lip landmark indices (mediapipe Face Mesh, 468-point): inner lip top/bottom
    # for the aperture, eye corners for a scale-invariant normaliser.
    _LIP_TOP, _LIP_BOT, _EYE_R, _EYE_L = 13, 14, 33, 263
    _LIP_ALPHA = 0.3  # EMA on lip motion: responsive (lips oscillate fast) but de-noised
    _LIP_GAIN = 15.0  # maps normalised lip-aperture deltas into ~0..1

    def _set_speaker(  # pragma: no cover
        self,
        frame,
        dets: list[Detection],
        speech: bool,
        prev_open: dict[int, float],
        lip_ema: dict[int, float],
    ) -> None:
        """Per-person active-speaker score from lip motion, gated by audio speech.

        mouth aperture (inner-lip gap / eye distance) tracked over time -> a mouth
        that *oscillates* is talking; one that's open-but-still (or invisible) is
        not. Gated by `speech` so silent mouth movement (chewing, nodding) doesn't
        count. Tracks whose face we can't mesh this frame decay toward 0."""
        import math

        import cv2

        h, w = frame.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gate = 1.0 if speech else 0.25
        seen: set[int] = set()

        for lm in res.multi_face_landmarks or []:
            L = lm.landmark

            def px(i, _L=L):
                return (_L[i].x * w, _L[i].y * h)

            tx, ty = px(self._LIP_TOP)
            bx, by = px(self._LIP_BOT)
            er, el = px(self._EYE_R), px(self._EYE_L)
            scale = math.hypot(er[0] - el[0], er[1] - el[1]) or 1.0
            openness = math.hypot(tx - bx, ty - by) / scale
            mcx, mcy = (tx + bx) / 2, (ty + by) / 2

            # attach the mesh to the person box whose region contains the mouth
            for d in dets:
                if d.cls_name == "person" and d.x1 <= mcx <= d.x2 and d.y1 <= mcy <= d.y2:
                    d.mouth_open = openness
                    tid = d.track_id
                    if tid is None:
                        d.lip_motion = 0.0
                        d.speaker_score = 0.0
                        break
                    prev = prev_open.get(tid)
                    move = abs(openness - prev) if prev is not None else 0.0
                    prev_open[tid] = openness
                    ema = lip_ema.get(tid, 0.0) * (1.0 - self._LIP_ALPHA) + move * self._LIP_ALPHA
                    lip_ema[tid] = ema
                    d.lip_motion = min(ema * self._LIP_GAIN, 1.0)
                    d.speaker_score = d.lip_motion * gate
                    seen.add(tid)
                    break

        # no lips meshed this frame -> let the talking signal fade, don't strand it
        for d in dets:
            if d.track_id is not None and d.track_id not in seen and d.track_id in lip_ema:
                lip_ema[d.track_id] *= 0.6
                d.lip_motion = min(lip_ema[d.track_id] * self._LIP_GAIN, 1.0)
                d.speaker_score = d.lip_motion * gate

    def _refine_faces(self, frame, dets: list[Detection]) -> None:  # pragma: no cover
        import cv2

        h, w = frame.shape[:2]
        res = self._face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return
        for face in res.detections:
            box = face.location_data.relative_bounding_box
            fcx = (box.xmin + box.width / 2) * w
            fcy = (box.ymin + box.height / 2) * h
            # attach the face to the person box that contains its center
            for d in dets:
                if d.cls_name == "person" and d.x1 <= fcx <= d.x2 and d.y1 <= fcy <= d.y2:
                    d.has_face = True
                    d.face_cx = fcx
                    d.face_cy = fcy
                    break
