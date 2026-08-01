"""Absent - surgical instrument count verification, fully on-device.

FastSAM segments the tray continuously so the feed looks live. It is
class-agnostic, so unlike COCO-trained YOLO it outlines instruments nobody
trained it on. It cannot name anything, which is Gemma's job, and Gemma only
fires on a count event, which is also how real surgical counts work.
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import FastSAM

import gemma

# Accepts a webcam index or a stream URL, so a phone-camera app works too.
_source = os.environ.get("ABSENT_CAMERA", "0")
CAMERA_INDEX = int(_source) if _source.isdigit() else _source

FRAME_W, FRAME_H = 1280, 720
SEG_SIZE = 640          # 448 was noisy and split single objects into fragments
CONF = 0.5

# FastSAM segments everything including the table and the background sheet.
# Keep only masks in a plausible instrument size range, as a fraction of frame.
MIN_AREA, MAX_AREA = 0.004, 0.20

# FastSAM happily returns the same object several times, plus slivers of it.
# Drop a mask that overlaps a bigger kept one, or that is mostly inside it.
MAX_IOU = 0.35
MAX_CONTAINED = 0.65

CAPTURES = Path("captures")
CAPTURES.mkdir(exist_ok=True)

app = FastAPI()
model = FastSAM("FastSAM-s.pt")

state = {"stage": "idle", "count_in": [], "present": [], "missing": [],
         "busy": False, "camera": CAMERA_INDEX, "tracking": 0, "source": "webcam"}
_lock = threading.Lock()
_raw = None
_annotated = None
_phone = None          # latest frame posted by the phone
_phone_seen = 0.0      # when it arrived, so we can fall back if the phone drops


def _palette(n):
    hues = np.linspace(0, 179, max(n, 1), dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 255), np.full_like(hues, 255)], axis=1)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)


def _instrument_masks(result, shape):
    """FastSAM outlines everything it can see, often several times over. Drop
    the background sheet, the table, speckle, and duplicates of one object."""
    if result.masks is None:
        return []
    h, w = shape[:2]
    frame_area = h * w

    sized = []
    for mask in result.masks.data:
        m = cv2.resize(mask.cpu().numpy(), (w, h)) > 0.5
        area = int(m.sum())
        if MIN_AREA <= area / frame_area <= MAX_AREA:
            sized.append((area, m))

    kept = []
    for area, m in sorted(sized, key=lambda p: -p[0]):
        duplicate = False
        for kept_area, k in kept:
            overlap = int(np.logical_and(m, k).sum())
            if not overlap:
                continue
            union = area + kept_area - overlap
            if overlap / union > MAX_IOU or overlap / area > MAX_CONTAINED:
                duplicate = True
                break
        if not duplicate:
            kept.append((area, m))
    return [m for _, m in kept]


def _overlay(frame, masks):
    out = frame.copy()
    colors = _palette(len(masks))
    tint = out.copy()
    for color, m in zip(colors, masks):
        tint[m] = color
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    for color, m in zip(colors, masks):
        contours, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color.tolist(), 2)

    cv2.putText(out, f"tracking {len(masks)}", (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return out


def _open(source):
    if isinstance(source, int):
        cam = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        return cam
    return cv2.VideoCapture(source)  # stream URL from a phone camera app


def _capture_loop():
    """Watches state['camera'] so the iPhone virtual webcam can be selected
    from the UI without restarting the server mid-demo."""
    global _raw, _annotated
    current = state["camera"]
    cam = _open(current)

    while True:
        # A phone posting frames takes over; if it stops for 3s we fall back
        # to the webcam so the demo can never end up on a frozen picture.
        with _lock:
            phone, age = _phone, time.time() - _phone_seen
        if phone is not None and age < 3.0:
            state["source"] = "phone"
            frame = phone
        else:
            state["source"] = "webcam"
            if state["camera"] != current:
                cam.release()
                current = state["camera"]
                cam = _open(current)
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue
        result = model.predict(frame, imgsz=SEG_SIZE, conf=CONF, verbose=False)[0]
        masks = _instrument_masks(result, frame.shape)
        with _lock:
            _raw = frame
            _annotated = _overlay(frame, masks)
            state["tracking"] = len(masks)


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


@app.get("/phone")
def phone_page():
    return FileResponse("static/phone.html")


@app.post("/ingest")
async def ingest(request: Request):
    """Frames posted by the phone's browser over the USB cable."""
    global _phone, _phone_seen
    data = await request.body()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False}
    with _lock:
        _phone = frame
        _phone_seen = time.time()
    return {"ok": True}


@app.get("/cameras")
def cameras():
    """Probe the first few indices so we can find whichever one the iPhone
    virtual webcam landed on."""
    found = []
    for i in range(5):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam.isOpened():
            ok, frame = cam.read()
            if ok:
                found.append({"index": i, "w": frame.shape[1], "h": frame.shape[0]})
        cam.release()
    return found


@app.post("/camera/{index}")
def set_camera(index: int):
    state["camera"] = index
    return state


@app.post("/source")
def set_source(url: str):
    """Point at a phone camera app's stream instead of a local webcam."""
    state["camera"] = url
    return state


@app.post("/reset")
def reset():
    state.update(stage="idle", count_in=[], present=[], missing=[])
    return state
