"""Absent - surgical instrument count verification, fully on-device.

FastSAM segments the tray continuously so the feed looks live. It is
class-agnostic, so unlike COCO-trained YOLO it outlines instruments nobody
trained it on. It cannot name anything, which is Gemma's job, and Gemma only
fires on a count event, which is also how real surgical counts work.
"""

import os
import re
import threading
from collections import Counter
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import FastSAM

import gemma

# Accepts a webcam index or a stream URL, so a phone-camera app works too.
_source = os.environ.get("ABSENT_CAMERA", "0")
CAMERA_INDEX = int(_source) if _source.isdigit() else _source

FRAME_W, FRAME_H = 1280, 720
# 448 was noisy, 640 dropped the feed to 4 fps, and FastSAM-x is 1.2 s a frame
# for no fewer masks. 512 with the cleanup below is the honest sweet spot.
SEG_SIZE = 512
CONF = 0.6

# FastSAM segments everything including the table and the background sheet.
# Keep only masks in a plausible instrument size range, as a fraction of frame.
MIN_AREA, MAX_AREA = 0.004, 0.20

# FastSAM happily returns the same object several times, plus slivers of it.
# Drop a mask that overlaps a bigger kept one, or that is mostly inside it.
MAX_IOU = 0.30
MAX_CONTAINED = 0.55

# Wispy background fragments have low solidity; real tools are compact.
MIN_SOLIDITY = 0.55
# Objects on a tray do not run off the edge of the frame. Wall and table
# fragments do, and they were the slivers cluttering the overlay.
MAX_BORDER = 0.30

# The count zone. Only what is inside this rectangle is tracked or counted, so
# the jacket standing in for the body defines the field and the rest of the room
# cannot wander into the count. Fractions of the frame, drag on the feed to move.
ROI = {"x": 0.34, "y": 0.30, "w": 0.32, "h": 0.40}

CAPTURES = Path("captures")
CAPTURES.mkdir(exist_ok=True)

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
model = FastSAM("FastSAM-s.pt")

state = {"stage": "idle", "count_in": [], "present": [], "missing": [],
         "busy": False, "camera": CAMERA_INDEX, "tracking": 0, "source": "webcam",
         "roi": dict(ROI), "error": "", "named": 0, "segmented": 0,
         "at_in": "", "at_out": ""}
_lock = threading.Lock()
_raw = None
_annotated = None
_masks = []
_phone = None          # latest frame posted by the phone
_phone_seen = 0.0      # when it arrived, so we can fall back if the phone drops

CATALOG = Path("static/catalog")
CATALOG.mkdir(parents=True, exist_ok=True)
catalog = []           # the manifest, one entry per instrument
events = []            # what happened and when, newest first
pending = Counter()    # names seen once, not yet trusted enough to catalogue

# The scene must hold still this many samples (at 0.25s each) before Gemma is
# asked anything, so a hand moving through the field never triggers a count.
# Gemma is a fixed ~3.7s no matter what we send it, so every millisecond of
# perceived latency has to come out of the settle time instead.
STABLE_SAMPLES = 8     # 0.1s apart, so the scene must hold for ~0.8s
STABLE_MAJORITY = 6    # of those 8, not all 8: masks flicker
WATCH_PERIOD = 0.4     # cooldown after a call returns
HEARTBEAT = 12.0       # re-check even when the scene looks unchanged
MANIFEST_CAP = 8       # hard stop against free-association on a dark crop

# Gemma is told to skip these and names them anyway. The jacket standing in for
# the body is the dangerous one: catalogue it and it "goes missing" the moment
# the camera shifts, which is a phantom retained instrument on the projector.
BLOCKLIST = {
    "hand", "hands", "finger", "fingers", "arm", "person", "people", "face",
    "jacket", "coat", "hoodie", "sweater", "shirt", "sleeve", "clothing",
    "cloth", "fabric", "pocket", "table", "surface", "desk", "floor", "wall",
    "background", "shadow", "light", "reflection", "line", "circle", "c",
    "hair", "head", "hairstyle", "skin", "neck", "ear", "eye", "nose", "mouth",
    "torso", "chest", "shoulder", "body",
}


def _palette(n):
    hues = np.linspace(0, 179, max(n, 1), dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 255), np.full_like(hues, 255)], axis=1)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)


def _roi(shape):
    h, w = shape[:2]
    r = state["roi"]
    x1, y1 = int(w * r["x"]), int(h * r["y"])
    return x1, y1, min(w, x1 + int(w * r["w"])), min(h, y1 + int(h * r["h"]))


def _in_roi(m, shape):
    """Overlap, not centroid. An item being pushed into the jacket has its
    centroid leave the rectangle well before the item is actually gone, and a
    hand entering from below counts the moment the wrist crosses the line."""
    x1, y1, x2, y2 = _roi(shape)
    area = int(m.sum())
    if not area:
        return False
    return int(m[y1:y2, x1:x2].sum()) / area >= 0.5


def _clean(m):
    """Smooth the speckle off a raw mask and keep only its largest blob, so one
    mask means one object rather than an object plus confetti."""
    m = cv2.morphologyEx(m.astype("uint8"), cv2.MORPH_OPEN, _KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _is_object(m, area):
    """Reject wispy background fragments and anything running off the frame."""
    h, w = m.shape
    border = m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum()
    if border / (2 * (h + w)) > MAX_BORDER:
        return False

    contours, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    hull = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return hull > 0 and area / hull >= MIN_SOLIDITY


def _instrument_masks(result, shape):
    """FastSAM outlines everything it can see, often several times over. Drop
    the background sheet, the table, speckle, and duplicates of one object."""
    if result.masks is None:
        return []
    # All the filtering runs at the mask's own resolution, not the frame's.
    # Doing it at 1280x720 cost ~3x the frame time for identical results.
    raw = result.masks.data.cpu().numpy() > 0.5
    mh, mw = raw.shape[1:]
    mask_area = mh * mw

    sized = []
    for m in raw:
        m = _clean(m)
        if m is None:
            continue
        area = int(m.sum())
        if MIN_AREA <= area / mask_area <= MAX_AREA and _is_object(m, area):
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

    h, w = shape[:2]
    full = [cv2.resize(m.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            for _, m in kept]
    return [m for m in full if _in_roi(m, shape)]


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

    x1, y1, x2, y2 = _roi(frame.shape)
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
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
    global _raw, _annotated, _masks
    current = state["camera"]
    cam = _open(current)
    last = time.time()
    misses = 0

    while True:
      # One unhandled throw in here used to kill the thread while /feed kept
      # serving the last frame forever, so a dead camera looked alive.
      try:
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
                # Unplugging the camera, or another app grabbing it, used to
                # spin here forever. Reopen instead, so replugging recovers.
                misses += 1
                if misses % 40 == 0:
                    cam.release()
                    cam = _open(current)
                time.sleep(0.05)
                continue
            misses = 0
        result = model.predict(frame, imgsz=SEG_SIZE, conf=CONF, verbose=False)[0]
        masks = _instrument_masks(result, frame.shape)
        now = time.time()
        state["fps"] = round(0.8 * state.get("fps", 0) + 0.2 / max(now - last, 1e-3), 1)
        last = now
        with _lock:
            _raw = frame
            _masks = masks
            _annotated = _overlay(frame, masks)
            state["tracking"] = len(masks)
      except Exception as e:
        state["error"] = f"camera: {type(e).__name__}"
        time.sleep(0.2)


def _snapshot(tag):
    """Gemma only ever sees the count zone. Cropping to it keeps the rest of
    the room, and whoever is standing in it, out of the inventory."""
    with _lock:
        frame = None if _raw is None else _raw.copy()
    if frame is None:
        raise RuntimeError("no frame yet")
    x1, y1, x2, y2 = _roi(frame.shape)
    zone = frame[y1:y2, x1:x2]
    stamp = datetime.now().strftime("%H%M%S")
    cv2.imwrite(str(CAPTURES / f"{tag}_{stamp}.jpg"), zone)
    return zone


@app.on_event("startup")
def startup():
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=gemma.warm_up, daemon=True).start()
    threading.Thread(target=_watch_loop, daemon=True).start()


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


@app.get("/frame.jpg")
def frame_jpg():
    """One frame, not a stream. An MJPEG <img> that drops dies permanently and
    leaves a black screen with no way back; polling single frames always
    recovers on the next request."""
    with _lock:
        frame = _annotated
    if frame is None:
        return Response(status_code=503)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return Response(status_code=503)
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/state")
def get_state():
    return state


@app.post("/count/in")
def count_in():
    state["busy"] = True
    try:
        seen = state["tracking"]
        items = gemma.inventory(_snapshot("count_in"))
        # Two independent signals: FastSAM counts shapes, Gemma names things.
        # They are wrong in different ways, so a disagreement is worth showing
        # rather than hiding. This is the cheapest guard against Gemma quietly
        # under-listing the tray at count-in.
        state.update(stage="counted_in", count_in=items, present=[], missing=[],
                     error="", named=len(items), segmented=seen,
                     at_in=datetime.now().strftime("%H:%M:%S"))
    except Exception as e:                      # never 500 mid-demo
        state["error"] = f"{type(e).__name__}: {e}"[:120]
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
        state.update(stage="counted_out", present=present, missing=missing, error="",
                     at_out=datetime.now().strftime("%H:%M:%S"))
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"[:120]
    finally:
        state["busy"] = False
    return state


@app.get("/try")
def try_page():
    return FileResponse("static/try.html")


@app.post("/identify")
def identify():
    """Hold something up, find out if Gemma can name it. Proof, not product."""
    with _lock:
        frame = None if _raw is None else _raw.copy()
    if frame is None:
        return {"name": "no camera frame yet"}
    x1, y1, x2, y2 = _roi(frame.shape)
    started = time.time()
    try:
        name = gemma.identify_held(frame[y1:y2, x1:x2])
    except Exception as e:                      # a 500 leaves the button dead
        name = f"gemma unavailable ({type(e).__name__})"
    return {"name": name, "seconds": round(time.time() - started, 1)}


def _held_mask(masks, shape):
    """The object being held up is the one nearest the middle of the count zone."""
    x1, y1, x2, y2 = _roi(shape)
    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
    best, best_d = None, None
    for m in masks:
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        d = (ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2
        if best_d is None or d < best_d:
            best, best_d = m, d
    return best


def _cutout(frame, mask):
    """Crop to the mask and fade everything around it, so a catalog card shows
    the instrument rather than whatever was behind it."""
    ys, xs = np.nonzero(mask)
    # np.ptp(arr), not arr.ptp(): numpy 2.0 removed the method and this ran on
    # every catalog add, so holding an item up and pressing add always 500'd.
    pad = int(0.08 * max(np.ptp(ys), np.ptp(xs)) + 8)
    y1, y2 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(frame.shape[1], xs.max() + pad)

    crop = frame[y1:y2, x1:x2].copy()
    m = mask[y1:y2, x1:x2]
    crop[~m] = (crop[~m] * 0.25).astype("uint8")
    return crop


@app.get("/catalog")
def catalog_page():
    return FileResponse("static/catalog.html")


@app.get("/catalog/items")
def catalog_items():
    return catalog


@app.post("/catalog/add")
def catalog_add():
    with _lock:
        frame = None if _raw is None else _raw.copy()
        masks = list(_masks)
    if frame is None:
        return {"ok": False, "error": "no camera frame yet"}

    x1, y1, x2, y2 = _roi(frame.shape)
    try:
        name = gemma.identify_held(frame[y1:y2, x1:x2])
    except Exception as e:                      # never 500 mid-demo
        return {"ok": False, "error": f"gemma: {type(e).__name__}"}
    if name in ("none", ""):
        return {"ok": False, "error": "hold the item up inside the count zone"}

    mask = _held_mask(masks, frame.shape)
    try:
        thumb = _cutout(frame, mask) if mask is not None else frame[y1:y2, x1:x2]
    except Exception:
        thumb = frame[y1:y2, x1:x2]             # a card without a cutout beats none
    fname = f"{len(catalog):02d}_{re.sub(r'[^a-z0-9]+', '-', name).strip('-')}.jpg"
    cv2.imwrite(str(CATALOG / fname), thumb)

    entry = {"name": name, "file": f"/static/catalog/{fname}",
             "at": datetime.now().strftime("%H:%M:%S"), "segmented": mask is not None,
             "status": "counted in"}
    catalog.append(entry)
    return {"ok": True, "item": entry, "count": len(catalog)}


def _watch_loop():
    """Autonomous count. Nobody presses anything: the scene is watched, and
    when it settles, Gemma is asked once what is there. Items that appear get
    added to the manifest, items that stop being visible are inside the body,
    items that come back out are accounted for again.

    Gemma costs ~4s a call so it cannot run per frame. Mask count is the cheap
    per-frame signal, and it only fires a call once that count has held still,
    which also keeps a hand moving through the field from poisoning anything.
    """
    history, last_run, last_count = [], 0.0, None

    while True:
        time.sleep(0.1)
        history.append(state["tracking"])
        history = history[-STABLE_SAMPLES:]
        if len(history) < STABLE_SAMPLES:
            continue

        # Not exact equality. Requiring 12 identical samples never fires while
        # a hand is in shot; the most common value winning 9 of 12 rides out
        # FastSAM dropping a mask for two or three frames.
        common = max(set(history), key=history.count)
        if history.count(common) < STABLE_MAJORITY or history[-1] != common:
            continue
        if state["busy"] or time.time() - last_run < WATCH_PERIOD:
            continue
        # Normally only re-ask when the number of things on screen changed, but
        # swapping one item for another keeps the count identical, so re-check
        # anyway on a slow heartbeat.
        if common == last_count and time.time() - last_run < HEARTBEAT:
            continue
        if common == 0 and not catalog:
            continue                       # empty scene, nothing to do

        time.sleep(0.15)                   # let the hand finish withdrawing
        last_run, dropped = time.time(), last_count is not None and common < last_count
        last_count = common
        try:
            _reconcile(dropped, can_add=common > 0)
        except Exception as e:
            state["error"] = f"watch: {type(e).__name__}"[:120]


def _reconcile(dropped, can_add):
    """One Gemma call, then fold what it saw into the manifest."""
    with _lock:
        frame = None if _raw is None else _raw.copy()
        masks = list(_masks)
    if frame is None:
        return

    x1, y1, x2, y2 = _roi(frame.shape)
    state["busy"] = True
    try:
        seen = Counter(gemma.inventory(frame[y1:y2, x1:x2]))
        state["error"] = ""
    finally:
        state["busy"] = False

    for item in catalog:
        # Match, not equality: Gemma answers "scissor" for "scissors", and a
        # rename is both a phantom new item and a phantom retained one.
        hit = gemma._match(item["name"], [n for n in seen if seen[n] > 0])
        if hit:
            seen[hit] -= 1
            item["misses"] = 0
            if item["status"] != "in view":
                item["status"] = "in view"
                _log(f"{item['name']} came back out")
        else:
            item["misses"] += 1
            # Asymmetric on purpose. If the mask count also dropped, geometry
            # corroborates the disappearance and it is called immediately.
            # Otherwise Gemma just under-reported, so wait for a second miss.
            if item["status"] != "inside" and (dropped or item["misses"] >= 2):
                item["status"] = "inside"
                _log(f"{item['name']} went inside")

    # Segmentation is the gatekeeper for new items. If it sees no shapes in the
    # zone, nothing can have arrived, whatever Gemma thinks it read in the
    # noise of an empty crop.
    if not can_add:
        return

    for name in set(seen.elements()):
        if len(catalog) >= MANIFEST_CAP or not _plausible(name):
            continue
        if gemma._match(name, [i["name"] for i in catalog]):
            continue                       # a rename of something we already hold
        _add_item(name, frame, masks)


def _plausible(name):
    """Keep the manifest to things that could actually be an instrument."""
    return (name not in BLOCKLIST
            and len(name.split()) <= 3
            and "part of" not in name and "piece of" not in name
            and not any(w in BLOCKLIST for w in name.split()))


def _add_item(name, frame, masks):
    """A name Gemma sees that the manifest does not have yet.

    The picture is the mask nearest the middle of the zone, because that is
    where a person deliberately holds the thing they want counted. Guessing
    which mask is "new" from stored centroids reliably picked the odd sliver
    at the edge of frame instead."""
    x1, y1, x2, y2 = _roi(frame.shape)
    zone = frame[y1:y2, x1:x2]
    mask = _held_mask(masks, frame.shape)
    try:
        # A mask covering most of the zone is the jacket or a torso, not a tool.
        if mask is not None and mask.sum() < 0.4 * max((y2 - y1) * (x2 - x1), 1):
            thumb = _cutout(frame, mask)
        else:
            thumb = zone
    except Exception:
        thumb = zone

    fname = f"{len(catalog):02d}_{re.sub(r'[^a-z0-9]+', '-', name).strip('-')}.jpg"
    cv2.imwrite(str(CATALOG / fname), thumb)
    catalog.append({"name": name, "file": f"/static/catalog/{fname}",
                    "at": datetime.now().strftime("%H:%M:%S"), "status": "in view",
                    "misses": 0})
    _log(f"{name} counted in")


def _log(message):
    events.insert(0, {"at": datetime.now().strftime("%H:%M:%S"), "what": message})
    del events[40:]


@app.get("/events")
def get_events():
    return events


@app.post("/verify")
def verify():
    """Check the manifest against what the camera can see right now. Anything
    catalogued but no longer visible is unaccounted for."""
    if not catalog:
        return {"ok": False, "error": "catalog an item first"}
    state["busy"] = True
    try:
        names = [i["name"] for i in catalog]
        state["error"] = ""
        present, missing = gemma.check_against(_snapshot("verify"), names)
        outstanding = Counter(missing)
        for item in catalog:
            gone = outstanding[item["name"]] > 0
            if gone:
                outstanding[item["name"]] -= 1
            item["status"] = "unaccounted for" if gone else "accounted for"
        state.update(stage="counted_out", count_in=names, present=present,
                     missing=missing, error="", named=len(names),
                     segmented=state["tracking"],
                     at_out=datetime.now().strftime("%H:%M:%S"))
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"[:120]
    finally:
        state["busy"] = False
    return {"ok": not state["error"], "items": catalog, "missing": state["missing"]}


@app.post("/catalog/clear")
def catalog_clear():
    catalog.clear()
    events.clear()
    pending.clear()
    for f in CATALOG.glob("*.jpg"):
        f.unlink(missing_ok=True)
    reset()
    return {"ok": True}


@app.get("/phone")
def phone_page():
    return FileResponse("static/phone.html")


@app.post("/ingest")
async def ingest(request: Request):
    """Frames posted by the phone's browser over the USB cable."""
    global _phone, _phone_seen
    data = await request.body()
    if not data:
        return {"ok": False, "error": "empty body"}   # imdecode raises on b""
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        # HEIC, the iPhone's default photo format, lands here.
        return {"ok": False, "error": "unsupported image format, use JPEG"}

    # The live stream caps its own size, the photo picker does not. A 12 MP
    # still would put ~1.5 s of overlay work inside the capture lock.
    big = max(frame.shape[:2])
    if big > 1280:
        s = 1280 / big
        frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
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
    state.update(stage="idle", count_in=[], present=[], missing=[], error="",
                 named=0, segmented=0, at_in="", at_out="")
    return state


@app.post("/zone")
async def set_zone(request: Request):
    """Drag a rectangle on the feed to say where the count happens."""
    r = await request.json()
    x = min(max(float(r["x"]), 0.0), 0.95)
    y = min(max(float(r["y"]), 0.0), 0.95)
    state["roi"] = {"x": x, "y": y,
                    "w": min(max(float(r["w"]), 0.05), 1.0 - x),
                    "h": min(max(float(r["h"]), 0.05), 1.0 - y)}
    return state
