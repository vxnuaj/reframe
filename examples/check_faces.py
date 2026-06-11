"""Quick empirical check: does YOLO find the people, and does full-range BlazeFace
find the faces, on real frames? Annotates 6 sampled frames (green = person,
red = face) and prints counts."""

import sys

import cv2
import mediapipe as mp
from ultralytics import YOLO

video = sys.argv[1] if len(sys.argv) > 1 else "assets/dwarkesh.mp4"
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
yolo = YOLO("yolo11n.pt")
face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

for k, frac in enumerate((0.05, 0.2, 0.4, 0.6, 0.8, 0.95)):
    i = int(total * frac)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    r = yolo(frame, classes=[0], conf=0.3, verbose=False)[0]
    persons = [b.xyxy[0].tolist() for b in (r.boxes or [])]
    res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    if res.detections:
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            faces.append((bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h))
    print(f"frame {i:>5}: persons={len(persons)}  faces={len(faces)}")
    for x1, y1, x2, y2 in persons:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    for ox, oy, fw, fh in faces:
        cv2.rectangle(frame, (int(ox), int(oy)), (int(ox + fw), int(oy + fh)), (0, 0, 255), 4)
    cv2.imwrite(f"out/check_{k}.jpg", frame)

cap.release()
print("wrote out/check_0..5.jpg")
