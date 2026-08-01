"""Absent - surgical instrument count verification, fully on-device.

YOLOv8 segments the tray continuously so the feed looks live. Gemma 4 only
fires on a count event, which is also how real surgical counts work.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO

import gemma

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
YOLO_SIZE = 448
CONF = 0.25

CAPTURES = Path("captures")
CAPTURES.mkdir(exist_ok=True)

app = FastAPI()
model = YOLO("yolov8n-seg.pt")

state = {"stage": "idle", "count_in": [], "present": [], "missing": [], "busy": False}
_lock = threading.Lock()
_raw = None
_annotated = None


def _palette(n):
    hues = np.linspace(0, 179, max(n, 1), dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 255), np.full_like(hues, 255)], axis=1)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)


def _overlay(frame, result):
    """Class-agnostic masks and boxes. We deliberately do not draw YOLO's own
    labels - naming is Gemma's job, and COCO labels would be wrong anyway."""
    out = frame.copy()
    if result.masks is None:
        return out

    colors = _palette(len(result.masks))
    tint = out.copy()
    for i, mask in enumerate(result.masks.data):
        m = cv2.resize(mask.cpu().numpy(), (out.shape[1], out.shape[0])) > 0.5
        tint[m] = colors[i]
    out = cv2.addWeighted(tint, 0.4, out, 0.6, 0)

    for i, box in enumerate(result.boxes.xyxy.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[i].tolist(), 2)

    cv2.putText(out, f"tracking {len(result.masks)}", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def _capture_loop():
    global _raw, _annotated
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cam.isOpened():
        raise RuntimeError(f"camera {CAMERA_INDEX} would not open")

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue
        result = model.predict(frame, imgsz=YOLO_SIZE, conf=CONF, verbose=False)[0]
        with _lock:
            _raw = frame
            _annotated = _overlay(frame, result)


def _snapshot(tag):
    with _lock:
        frame = None if _raw is None else _raw.copy()
    if frame is None:
        raise RuntimeError("no frame yet")
    stamp = datetime.now().strftime("%H%M%S")
    cv2.imwrite(str(CAPTURES / f"{tag}_{stamp}.jpg"), frame)
    return frame


@app.on_event("startup")
def startup():
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=gemma.warm_up, daemon=True).start()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/feed")
def feed():
    def frames():
        while True:
            with _lock:
                frame = _annotated
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            time.sleep(0.04)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/state")
def get_state():
    return state


@app.post("/count/in")
def count_in():
    state["busy"] = True
    try:
        items = gemma.inventory(_snapshot("count_in"))
        state.update(stage="counted_in", count_in=items, present=[], missing=[])
    finally:
        state["busy"] = False
    return state


@app.post("/count/out")
def count_out():
    if not state["count_in"]:
        return state
    state["busy"] = True
    try:
        present, missing = gemma.check_against(_snapshot("count_out"), state["count_in"])
        state.update(stage="counted_out", present=present, missing=missing)
    finally:
        state["busy"] = False
    return state


@app.post("/reset")
def reset():
    state.update(stage="idle", count_in=[], present=[], missing=[])
    return state
