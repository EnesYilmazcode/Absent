# Absent, full technical report

For judges, and for us. Written 3:20 PM on build day. Everything here was read out of the
code or measured on this laptop. Where a thing is not measured, it says so.

---

## DO THIS BEFORE YOU PRESENT

Set one environment variable in the same PowerShell window you launch the server from:

```powershell
$env:YOLO_OFFLINE = "1"
```

**Why it matters.** Ultralytics phones home. On import it does a DNS lookup for
`one.one.one.one`, and after every `predict` it POSTs to
`www.google-analytics.com/mp/collect` from a background thread. Both were observed live on
this machine, not assumed. Our track is On-Device Private Health and our README currently
says Absent "makes no network calls of any kind". With that variable set, the measured
count of outbound calls is **zero** through import, model load and predict. Without it,
that README sentence is false and a judge with a packet capture can prove it.

Verify it took:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics.utils.events import events; print(events.enabled)"
```

It must print `False`. Set the variable **before** starting the server, because the
analytics gate is evaluated once at import.

Two more one-line hardening items:

- `app.py:61` is `FastAPI()`, which auto-enables `/docs` and `/redoc`. Those pull Swagger
  and Redoc from `cdn.jsdelivr.net`. They never fire unless somebody opens those pages, but
  `FastAPI(docs_url=None, redoc_url=None)` removes the only CDN reference in the app.
- Soften the README line to "no cloud inference, and no image ever leaves the machine",
  which is true either way.

---

## Does it work offline?

**Yes, with two caveats worth stating plainly rather than hiding.**

What is proven:

- The only network call our own code makes is to `127.0.0.1:11434`, a literal loopback
  address. Every inference goes there. It survives wifi being off.
- The HTML has zero external references. No CDN, no web fonts, no remote images. Fonts are
  the system stack. Every `fetch()` is a relative path to our own server.
- The weights are on disk. `FastSAM-s.pt` is 23.8 MB in the repo root and `app.py:63`
  loads it by filename.
- Instrumenting `socket.getaddrinfo`, `socket.connect` and `urllib.request.urlopen`, then
  importing the whole app, produced one DNS lookup and zero TCP connections. With
  `YOLO_OFFLINE=1`, zero of both.

The two honest caveats:

1. **The phone path needs a link, not the internet.** Over USB the tether is still a
   network interface. If asked "is any network involved", the answer is yes, a cable
   between two devices you are holding. If asked "does anything reach the internet", the
   answer is no. Say the second only if you were asked the second.
2. **Ollama is a separate process and we did not audit its traffic.** Inference is local
   and images reach it over loopback, so no image leaves the machine through Absent. We
   cannot claim the Ollama process itself sends zero packets.

**How to prove it on stage.** Disable the wifi adapter itself, not airplane mode. Run
`Get-NetAdapter` with the window visible and let the room read `Status: Disabled`. A
software toggle is waveable; a disabled adapter is not. Then `ping 8.8.8.8`, say nothing
while it times out, and run a full count. Use the built-in webcam for this beat, because it
needs no network at all rather than merely no internet.

**Do not open Wireshark.** Windows does its own background connectivity checks and they
will muddy the capture with traffic that is not yours.

---

## How a day with Absent actually goes

1. You start Ollama and launch the server from the repo root. There is no launch script in the repo, so the command is typed by hand. The README's `python -m uvicorn app:app --port 8000` is stale: the running instance answers on HTTPS port 8443 with cert.pem and key.pem, which the phone's browser requires before it will hand over a camera. FEATURES.md already flags this as a known doc bug.

2. On startup, app.py fires three daemon threads: the capture loop, a Gemma warm-up call, and the watch loop (app.py:380-384). The warm-up sends a 64x64 black square to Ollama so a cold-start crash cannot happen on stage, and keep_alive -1 pins the model in memory afterward (gemma.py:74-75).

3. You open the laptop page at /. There are no count buttons anymore. There is one Reset button and a sidebar that says Watching. The docs still describe Count In and Count Out buttons. The code does not have them. Trust the code.

4. You open /phone on the iPhone over the USB cable or hotspot. The page asks for the rear camera and starts streaming with no button press (phone.html:95). It also draws the count zone rectangle on the phone screen, polled from the laptop every 1.5s, so whoever holds the camera can aim (phone.html:69-78).

5. You drag a rectangle on the laptop video to place the count zone over the jacket standing in for the patient. That posts fractions to /zone (app.py:841). Only what sits inside that rectangle can be added to the manifest. Everything anywhere in frame still counts for presence tracking, which is how setting an instrument aside on the table does not raise an alarm.

6. Nothing is happening yet. The capture loop is pulling phone frames, segmenting each one with FastSAM, and painting colored outlines. Measured on the running server just now: 7.4 fps. The watch loop is sampling the in-zone mask count every 100ms and doing nothing, because the scene has not moved.

7. You bring an instrument into the count zone. The count zone changes frame to frame, _measure_motion reports above 0.06, and the watch loop marks the scene stirred (app.py:582).

8. You hold still. The watch loop waits for two things at once. The mask count has to be the same value in at least 6 of the last 8 samples (about 0.8s of stillness), and motion has to fall back under 0.05. If the camera is handheld and never settles, a 2.5s patience timer fires the call anyway rather than never asking (app.py:588-600).

9. Both conditions pass and the mask count went up, so something arrived. Before spending a Gemma call, the code checks whether that shape is already on the manifest by position and area (app.py:640-642). One fork read as fork and then as utensil would otherwise become two forks.

10. It is new. The count zone is cropped out of the raw frame, upscaled so its short side is at least 768 pixels, JPEG encoded, base64'd, and posted to Ollama at 127.0.0.1:11434 with think=false, temperature=0, seed=42. The prompt asks for one thing: name the single object the person is holding up. Measured live on this box today, twice: 1.6s and 1.4s.

11. The name comes back. Three filters run before it is trusted. A shape hint overrides the name to fork if the outline is more than 3 times longer than it is wide, because Gemma flips between fork, knife and spoon on exactly that silhouette (app.py:676-686). A blocklist of about 50 words throws out hands, jackets, shirts, tables and none (app.py:118-128). A fuzzy match against the existing manifest throws out a re-read of something already counted (app.py:651).

12. It survives. The mask nearest the middle of the zone is cut out, everything around it faded to 25 percent, and written to static/catalog/00_name.jpg. A card appears in the sidebar within 400ms with a real photo of the real object, its name and a timestamp. The event log records the item counted in. Nobody pressed anything.

13. You repeat for each instrument. A 3 second cooldown between adds enforces one object at a time, and the manifest is hard capped at 8 items so a dark crop cannot free-associate a twentieth entry.

14. Now the live moment. You hand the field to a judge and they push an instrument into the jacket, or pocket it. Gemma is not involved in this step at all. Every frame, _track_presence looks for a blob of about the right area within a quarter of the frame diagonal of where that item was last seen (app.py:155-191).

15. It finds nothing. Two seconds later the item's status flips to inside, the event log says it went inside, the sidebar card turns red and reads inside the patient, and the banner reads 1 unaccounted for followed by the name. End to end from the instrument disappearing to the red banner: 2 seconds plus one 400ms poll.

16. The judge puts it back. The blob reappears near where it was, the status flips back, the log says it came back out, the banner goes green. Reset clears the manifest, the cards, the event log and the thumbnail files.


## The pipeline, stage by stage

```
                    iPhone (Safari)
                          |
   getUserMedia -> <video> -> canvas -> JPEG q0.8, max side 1280
                          |
                          |  HTTPS POST /ingest, raw JPEG body
                          |  every ~80ms, self-clocking
                          v
  =========================== LAPTOP, one Python process ===========================

   [uvicorn event loop]     POST /ingest -> cv2.imdecode -> BGR ndarray
                                  |            (app.py:782)
                                  v  under _lock
                             _phone + _phone_seen
                                  |
                                  |  read under _lock; older than 3s ->
                                  |  fall back to the local webcam
                                  v
   [CAPTURE THREAD, CPU, ~7.4 fps measured]
        |
        +--> _measure_motion  96x96 grey absdiff -> state['motion']   (app.py:194)
        |
        +--> FastSAM-s  imgsz 512, conf 0.6, CPU torch -> mask stack  (app.py:344)
        |        |
        |        +--> _clean, _is_object, IoU dedupe -> clean masks   (app.py:240)
        |                 |                  |
        |                 |                  +--> in-zone subset -> _masks
        |                 |                            -> state['tracking']
        |                 |
        |                 +--> _track_presence (ALL masks, geometry, NO Gemma)
        |                            |                              (app.py:155)
        |                            +--> unmatched > 2.0s -> status = 'inside'
        |                            +--> matched again    -> status = 'in view'
        |                            +--> _log(...) -> events
        |
        +--> _overlay -> _annotated (painted BGR frame)               (app.py:279)

   [WATCH THREAD, ticks every 100ms]                                  (app.py:558)
        samples state['tracking'] and state['motion']
        |
        gate 1: motion > 0.06 marks the scene stirred
        gate 2: same mask count in 6 of the last 8 samples (~0.8s)
        gate 3: motion back under 0.05, or 2.5s of patience expired
        gate 4: not busy, 0.4s cooldown, count went UP, 3s since last add,
                manifest under 8, and the shape is not already catalogued
        |
        v  ROI crop only, upscaled to >=768px short side, JPEG, base64
   ----------------------------- HTTP 127.0.0.1:11434 -----------------------------
   [OLLAMA, separate process]  gemma4:e2b-it-qat
        think=false, temperature=0, seed=42, keep_alive=-1
        1.46 GB of 3.93 GB resident in VRAM (measured), the rest on CPU
        prompt: "name the single object the person is holding up"
        measured 1.6s and 1.4s live today
   ---------------------------------------------------------------------------------
        |
        v  one lowercase string
   _shape_hint -> BLOCKLIST -> fuzzy match vs manifest                (app.py:676)
        |
        v  survives
   _cutout -> static/catalog/NN_name.jpg  +  catalog.append({name, area, cx, cy})
        |                                                            (app.py:697)
        v
   [FASTAPI THREADPOOL]  GET /frame.jpg (JPEG, ~90ms poll)
                         GET /catalog/items + GET /state (JSON, 400ms poll)
        |
        v
   LAPTOP BROWSER  cards, red "inside the patient", banner "N unaccounted for"

  KEY: Gemma runs ONCE per instrument, at count-in, to answer "what is this".
       The "unaccounted for" alert is 100% geometry and never calls Gemma.
```

| Stage | In | Out | Where it runs | Latency | Code |
|---|---|---|---|---|---|
| Phone camera capture | Photons on the iPhone rear sensor | A live MediaStream inside Safari | iPhone, Safari, GPU-backed camera pipeline. Not the laptop. | not measured | static/phone.html:82-84 |
| Frame grab and JPEG encode on the phone | Video element pixels | JPEG blob, quality 0.8, longest side capped at 1280 | iPhone, Safari main thread, canvas 2D context. Loops on an 80ms setTimeout, so it is a self-clocking chain and never a fixed frame rate. | not measured | static/phone.html:50-66 |
| Upload to the laptop | JPEG blob | HTTPS POST body to /ingest | USB cable or hotspot link, TLS with the self-signed cert from make_cert.py | not measured | static/phone.html:45-48 |
| Ingest and decode | Raw JPEG bytes | BGR uint8 ndarray stored in the module global _phone, plus a timestamp | Server process, uvicorn event loop thread. This handler is async def, so cv2.imdecode runs on the event loop and blocks it for the duration. | not measured | app.py:782-803 |
| Source selection | _phone plus its age, or the local webcam | One BGR frame for this iteration | Server process, capture thread | Phone frames win. If none arrive for 3 seconds it falls back to the webcam, so the demo cannot end on a frozen picture. | app.py:321-342 |
| Motion measurement | The count zone crop of the frame | state['motion'], a float 0 to 1 | Server process, capture thread, CPU. One 96x96 greyscale absdiff against the previous frame. No model involved. | Negligible next to segmentation. Measured baseline on a still webcam scene right now: 0.02 to 0.047, against a trigger of 0.06. | app.py:194-210 |
| FastSAM segmentation | Full BGR frame | A stack of float mask logits at the model's own resolution | Server process, capture thread, CPU-only torch by design so all VRAM stays free for Gemma. Ultralytics releases the GIL for most of this. | The whole capture iteration measured live at 7.3 to 7.5 fps, so roughly 135ms per frame end to end. Segmentation is the bulk of it. imgsz is 512, chosen because 640 dropped the feed to 4 fps and FastSAM-x cost 1.2s a frame for no extra masks. | app.py:344, constants at app.py:33-34 |
| Mask cleanup and deduplication | Raw mask stack thresholded at 0.5 | A list of boolean masks resized to frame resolution, one per plausible object | Server process, capture thread, CPU, numpy and OpenCV. All filtering runs at mask resolution, not 1280x720, which the comment says cost 3x the frame time for identical results. | included in the 135ms above | app.py:240-276, with _clean at app.py:213-222 and _is_object at app.py:225-237 |
| Count zone membership | Each mask plus the ROI rectangle | The in-zone subset, stored as _masks | Server process, capture thread. Overlap based, not centroid based, so an item being pushed into the jacket does not leave the zone early. | negligible | app.py:144-152, called at app.py:352 |
| Presence tracking | All masks anywhere in frame, plus each catalog entry's stored area and centroid | Each catalog item's status flipped between 'in view' and 'inside', plus event log lines | Server process, capture thread, pure numpy geometry. No model, no Gemma. This is the stage that produces the alert. | Runs every frame, roughly every 135ms. An item must go unmatched for 2.0 seconds before it flips to inside. | app.py:155-191, called at app.py:353 |
| Overlay render | Frame plus masks | Annotated BGR frame with tinted fills, contours and the white zone rectangle | Server process, capture thread, CPU. Runs while holding _lock, together with the state assignments. | not measured separately | app.py:279-294, called at app.py:358 |
| Watch loop trigger decision | state['tracking'] sampled every 100ms, plus state['motion'] | A yes or no on spending a Gemma call | Server process, its own daemon thread. Independent of the capture thread and of every HTTP request. | Ticks at 100ms. Needs 8 samples, so an 0.8s stability window minimum. Falls through on a 12 second heartbeat even when nothing moves. | app.py:558-614 |
| Duplicate guard before the call | The mask nearest the centre of the zone, plus the catalog | Early return if that shape is already on the manifest | Server process, watch thread, geometry only | negligible | app.py:660-673, called at app.py:640 |
| Gemma naming | Count zone crop, upscaled so the short side is at least 768px, JPEG, base64 | One lowercase string, one to three words, or 'none' | Server process posts to Ollama at 127.0.0.1:11434 over a reused requests.Session. Ollama is a separate process. Measured just now via /api/ps: 1.46 GB of a 3.93 GB model resident in VRAM, so roughly 37 percent GPU offload and the rest on CPU. | Measured live today on the actual ROI crop: 1.6s and 1.4s. The 3.7 to 4.5s figure in README.md and CONTEXT.md was measured on a full 1280x720 frame with the harder inventory prompt, which this path no longer uses. | gemma.py:165-167, via gemma.py:66-80 and gemma.py:50-63 |
| Name validation | Gemma's raw string plus the candidate mask | Accept or discard | Server process, watch thread, CPU | negligible | shape hint app.py:676-686, blocklist app.py:689-694 against app.py:118-128, manifest fuzzy match gemma.py:110-123 |
| Catalog write | Accepted name, raw frame, mask | A faded cutout JPEG on disk under static/catalog/, plus a dict appended to the in-memory catalog list with name, file, time, status, area, cx, cy, seen | Server process, watch thread, disk write | not measured | app.py:697-725, cutout at app.py:501-514, held-mask selection at app.py:486-498 |
| Video delivery to the laptop UI | _annotated ndarray | A single JPEG at quality 80 per request | Server process, FastAPI threadpool thread. The main page polls single frames rather than using MJPEG, because a dropped MJPEG img element goes black permanently with no way back. | The browser re-requests 90ms after each image loads. The producer runs at about 7.4 fps, so some polls return the same frame. | app.py:408-421, browser side static/index.html:142-148 |
| Manifest and status delivery | catalog list and state dict | JSON over two fetches, rendered as cards, a banner and a cross-check line | Server process threadpool, then the laptop browser main thread | Polled every 400ms. That poll is the last link in the alert chain. | app.py:522-524 and app.py:424-426, browser side static/index.html:95-131 and :173 |

### Where data crosses a boundary

- iPhone camera to Safari: MediaStream, GPU texture, never touches the laptop.
- Safari video element to canvas: raw RGBA pixels in the phone's memory, downscaled to a 1280 longest side.
- Canvas to network: JPEG bytes at quality 0.8, wrapped in a Blob.
- Phone to laptop: HTTPS POST with Content-Type image/jpeg and the JPEG as the raw request body. Not multipart, not base64. TLS with a self-signed cert covering 172.20.10.x (Apple USB tethering) and 192.168.137.x (Windows Mobile Hotspot).
- Event loop to capture thread: cv2.imdecode turns the JPEG into a BGR uint8 ndarray, handed over through the module global _phone under threading.Lock (app.py:800-802). The capture thread reads it under the same lock (app.py:321-322).
- Capture thread to Ultralytics: the BGR ndarray goes straight into model.predict. Ultralytics does its own letterbox to 512 internally.
- Torch to numpy: result.masks.data.cpu().numpy() > 0.5, a boolean array at mask resolution, not frame resolution (app.py:247).
- Capture thread to everyone else: three globals under one lock. _raw (clean BGR frame), _masks (in-zone boolean masks), _annotated (painted BGR frame).
- Capture thread to watch thread: nothing direct. The watch thread reads state['tracking'], an int, and state['motion'], a float. That is deliberately the cheapest possible handoff.
- Watch thread to Ollama: the ROI crop is upscaled with INTER_CUBIC, JPEG encoded in memory, base64 ASCII encoded, and put in a JSON body's images array. Python to a separate Ollama process over HTTP on 127.0.0.1, never localhost, because on Windows localhost resolves to ::1 first and Ollama binds IPv4 only, which was costing a fixed 2.04s per call.
- Ollama back to Python: JSON, with the answer inside the response string. That string often has a markdown fence around its JSON, stripped by a regex at gemma.py:83-95.
- Watch thread to disk: a cutout JPEG written by cv2.imwrite into static/catalog/, which is also the directory FastAPI serves statically. The browser fetches it back over HTTP by URL, so the disk is the handoff.
- Server to laptop browser, video: JPEG bytes with Cache-Control no-store, one per request, cache-busted with a timestamp query string. The /feed MJPEG multipart route still exists and is used by /catalog and /try, but the main page does not use it.
- Server to laptop browser, data: two JSON fetches every 400ms, the catalog array and the state object.
- Server to phone browser: the state JSON, of which the phone reads only the roi fractions, so it can draw the same rectangle the laptop is using.

### State

- All state is in-memory Python globals in one process. There is no database, no file-backed store, nothing that survives a restart. The only things on disk are the cutout thumbnails in static/catalog/ and the debug snapshots in captures/.
- state (app.py:65-68) is a plain dict of scalars for the UI: stage, tracking, visible, motion, fps, busy, error, camera, source, roi, last_add. It is written by the capture thread, the watch thread and by request handlers, with no lock. GET /state returns it whole.
- catalog (app.py:80) is the manifest and the real product. Each entry holds name, thumbnail path, time, status, and the geometry presence tracking runs on: area, cx, cy, seen. It is appended by the watch thread in _add_item and mutated every frame by _track_presence in the capture thread.
- events (app.py:81) is the audit log, newest first, hard trimmed to 40 entries at app.py:730. Written only by _log.
- _raw, _masks, _all_masks, _annotated, _phone and _phone_seen are the frame pipeline globals, and they are the only state guarded by _lock.
- _prev_zone holds one 96x96 greyscale image for the motion difference.
- The watch loop holds state nothing else can see: history, last_run, last_count, stirred and stirred_at are all locals inside _watch_loop's stack frame.
- pending (app.py:82) is declared and cleared but never written to. It is dead code left over from an earlier two-sighting confirmation design.
- Item status is a string with four possible values across the codebase, and they do not all agree. The autonomous path writes 'in view' and 'inside'. The manual /catalog/add path writes 'counted in'. /verify writes 'accounted for' and 'unaccounted for'. index.html only paints a card red when the status is exactly 'inside'.
- Reset (POST /catalog/clear, app.py:766) clears the catalog, the events and the pending counter, deletes every thumbnail file, and calls reset() to blank the count fields. It does NOT reset the watch loop's local last_count, because that variable lives inside another thread's stack frame. That has a real consequence, listed under surprises.

### Things a judge will poke at

- The alert does not use Gemma. This is the thing a judge will find most surprising, so say it first rather than hide it. Gemma is asked exactly one question, once per instrument, when the instrument arrives: what is this called. Deciding that an instrument has gone missing is pure geometry in _track_presence, comparing mask area and centroid every frame. The honest answer is a good one: a vision model at 1.5 seconds a call cannot run per frame, it has no memory across calls so it cannot hold object identity, and geometry does that job better and ten times faster. Gemma is load-bearing for the part no detector can do without a labeled dataset.
- The docs describe a product that no longer exists. README.md and CONTEXT.md describe Count In and Count Out buttons, whole-frame inventory calls, and a set difference. index.html has one button, Reset. The /count/in, /count/out and /verify routes are still in app.py and still work, but nothing in the shipped UI calls them, so gemma.inventory and gemma.check_against are dead in the demo path. FEATURES.md's whole Flow A versus Flow B decision section is also stale, because the last four commits built a third flow that is neither one.
- The measured latency in the docs is for a call the app no longer makes. README says 3.7 to 4.5 seconds. That was the inventory prompt on a full 1280x720 frame. The live path sends only the ROI crop with the single-object prompt, and I measured it on the running server twice today at 1.6s and 1.4s. The real number is better than the claimed one, which is a good problem, but do not say four seconds on stage.
- Presence tracking flaps, and it is flapping right now. The running server's event log holds 7 went-inside and 7 came-back-out entries over 4 minutes for one catalogued phone that never left the table. Some pairs are 1 to 4 seconds apart. On a projector that means the card and the banner flip red and green while nothing is happening. The cause is FastSAM dropping that object's mask, or resizing it past AREA_TOLERANCE, for longer than the 2.0 second HIDDEN_AFTER window. If there is time for one fix before the lock, raise HIDDEN_AFTER and loosen AREA_TOLERANCE. A flapping alarm is the most damaging thing that can happen on stage.
- The motion trigger has almost no margin. MOTION_TRIGGER is 0.06 and MOTION_SETTLED is 0.05. I sampled the idle scene three times just now and got 0.031, 0.047 and 0.02. Idle noise on a handheld camera sits right under the trigger. The code already knows this, which is why MOTION_PATIENCE exists to fire the call anyway after 2.5 seconds, and why the comment at app.py:97 records that 0.01 for settled was unreachable and blocked every call.
- Reset does not fully reset. POST /catalog/clear empties the catalog, the events and the thumbnails, but the watch loop's last_count is a local variable in a different thread's stack frame and nothing touches it. New items are only added when the mask count goes up from last_count. So if you press Reset while two instruments are sitting in the zone, last_count stays at 2, the scene stays at 2, and nothing gets re-catalogued until you physically remove an item and put it back. Workaround for the demo: clear the zone first, then press Reset.
- The manual /catalog/add path can freeze the video feed. Entries created by POST /catalog/add (app.py:551) have no cx, cy, area or seen keys, but _track_presence reads item['cx'] unconditionally on every frame (app.py:177). That raises a KeyError inside the capture loop at line 353, before the block that assigns _annotated at line 354. The broad handler at app.py:361 catches it so the thread survives, but the picture stops updating forever and the only symptom is 'camera: KeyError' in the corner. Do not open /catalog during the demo. The autonomous path is unaffected because _add_item writes all four keys.
- Setting an instrument aside is handled, and it is a genuinely good detail worth mentioning out loud. _track_presence is passed every mask in the frame, not just the in-zone ones (app.py:353), while adding to the manifest only ever looks at the in-zone masks (app.py:627). So moving a counted instrument out of the count zone and onto the table keeps it accounted for. Only actually disappearing raises the alarm. This is the newest commit in the repo and it is documented nowhere except the code.
- A shape hint silently overrides the model. If Gemma answers fork, knife, spoon or utensil, the code measures the outline and rewrites the answer to fork whenever the object is more than 3 times longer than it is wide (app.py:676-686). That is a defensible fix for a known failure mode, but it is a hardcoded override of the model's output and a sharp judge may ask about it. Have the answer ready: the outline is the more reliable signal on that specific silhouette.
- The blocklist is doing heavy lifting and it barely holds. I triggered a real identify call on the live server a minute ago and got back 'a shirt'. That exact string is not in BLOCKLIST, but _plausible also checks each word individually (app.py:694), and shirt is on the list, so it was rejected. The second call returned 'card', which is not blocklisted at all and would go straight onto the manifest. On a dark crop the model will name things that are not instruments, and the only things standing between that and the manifest are a 50-word blocklist, a 3-word length limit, and an 8-item cap.
- Everything mutable except the frame buffers is unsynchronized. state and catalog are written by the capture thread, the watch thread and FastAPI's request threadpool with no lock. CPython's GIL makes this safe from corruption in practice, but /catalog/items can serialize the list on one thread while _add_item appends on another, and _track_presence mutates entries while the JSON encoder walks them.
- POST /ingest is async def, so cv2.imdecode of every incoming phone frame runs on the uvicorn event loop rather than the threadpool. At roughly a dozen posts a second that is a JPEG decode on the event loop a dozen times a second, competing with every other request. It has not caused a visible problem, but it is the wrong thread for that work.
- A fresh clone cannot run. .gitignore excludes *.pt, so FastSAM-s.pt is not in the repo, and there is no requirements.txt and no LICENSE file even though the README badge claims AGPL-3.0. Judges are told the repo is the source of truth. FEATURES.md already lists all three as outstanding.
- The whole system is one process on one machine, and that is checkable on stage. Ollama is bound to 127.0.0.1, the segmentation weights are a local file, and the web pages load no fonts, no CDN scripts and no remote images. Turning wifi off changes nothing, which is the privacy claim made testable rather than asserted.

---

## Stack

### Models

- gemma4:e2b-it-qat (Gemma 4 E2B, instruction-tuned, quantization-aware trained). 4.3 GB on disk (4,336,358,185 bytes), 4.6B parameters, Q4_0. Job: it is the only part of the system that knows what an object is called. At count-in it names every object in the count zone as a JSON array. At count-out it takes the count-in list in the prompt and says which of those names it can still see. It also answers the single-object 'what am I holding' call. Runs locally under Ollama 0.32.5, called over POST http://127.0.0.1:11434/api/generate with the frame as base64 JPEG. Verified loaded right now via /api/ps: 3.93 GB resident, 1.46 GB of that in VRAM, so roughly 37 percent GPU and the rest on CPU. Ollama reports its capabilities as completion, tools, thinking, vision. Called with think=false, temperature=0, seed=42, keep_alive=-1, context 4096.
- FastSAM-s (Ultralytics FastSAM, small). Local weights file FastSAM-s.pt, 23.9 MB, full precision, no quantization. Job: class-agnostic segmentation on every frame. It outlines objects without knowing their names, which is what makes the feed look live between count events. It is not COCO-limited, so it outlines instruments nobody trained it on. Runs on CPU on purpose, at 512 pixel inference size, confidence 0.6, under CPU-only torch 2.13.0+cpu with torch.cuda.is_available() confirmed False. All 4 GB of VRAM is left for Gemma.
- gemma4:e4b-it-qat is pulled and on disk (6.1 GB, 7.5B parameters, Q4_0) but is NOT used. It crashes llama-server on this card. See rejected.
- gemma3:4b-it-q4_K_M is on disk (3.3 GB, 4.3B, Q4_K_M) but is NOT used. Ollama lists its capabilities as completion only, with no vision, so it cannot see an image.
- yolov8n-seg.pt (7.1 MB), FastSAM-x.pt (145 MB), mobile_sam.pt (40.7 MB) and yoloe-11s-seg-pf.pt (27.9 MB) are downloaded in the folder from the segmentation bake-off. None of them are loaded by app.py. Only FastSAM-s.pt is.

### Frameworks

- Ollama 0.32.5 (version confirmed live from the running server). The local model server. It holds Gemma 4 E2B in memory and splits it between GPU and CPU. We call its /api/generate endpoint directly rather than the OpenAI-compatible one, because base64 images in the images field is what we tested and what works.
- Ultralytics 8.4.115. Loads and runs FastSAM-s for the per-frame segmentation masks.
- torch 2.13.0+cpu and torchvision 0.28.0+cpu. The CPU-only build, installed deliberately from the PyTorch CPU wheel index so segmentation never competes with Gemma for the 4 GB of VRAM.
- OpenCV (opencv-python) 5.0.0.93. Camera capture, all the mask cleanup (morphology, connected components, convex hull solidity), the colored overlay, JPEG encode for the feed, and the frame resize before it goes to Gemma.
- NumPy 2.4.4. The mask arrays and all the geometry: centroids, areas, overlap. Note it is NumPy 2.x, which removed ndarray.ptp(), and that broke the catalog cutout until it was changed to np.ptp().
- FastAPI 0.141.1 on Starlette 1.3.1. The whole app is one file of HTTP endpoints: the video feed, the count buttons, the phone frame ingest, the zone drag.
- Uvicorn 0.52.0 (with httptools 0.8.0, h11 0.16.0, websockets 17.0.1, watchfiles 1.2.0). The server. Runs over HTTPS with the self-signed cert so the phone will hand over its camera.
- Requests 2.34.2 with urllib3 2.7.0. One persistent requests.Session to Ollama, so the connection is reused instead of dialed again on every count.
- cryptography 50.0.0. make_cert.py uses it to generate a self-signed cert covering localhost, every local IP, Apple's 172.20.10.x USB tether range and the Windows hotspot 192.168.137.x range. Safari only gives a page the camera over HTTPS, and there is no internet at demo time to get a real certificate.
- Pydantic 2.13.4, Pillow 12.2.0, matplotlib 3.11.1. Transitive dependencies of FastAPI and Ultralytics, not used directly.
- Python 3.13.1, in a local .venv in the project folder.
- Frontend: plain HTML, CSS and JavaScript in static/ (index.html, try.html, catalog.html, phone.html). No framework, no CDN, no web fonts, no remote images. That is a privacy requirement, not a style choice. The page has to keep working with the wifi off.

### Why each choice

- Gemma instead of a fine-tuned detector: a detector only knows the classes somebody labeled for it, and Gemma names things it has never been trained on, so we skipped the dataset entirely.
- E2B instead of E4B: the card has 4 GB of VRAM, and E4B crashes on it.
- QAT quantization: it is the smallest build that still holds up, which is what fits on a 4 GB card.
- Ollama instead of running llama.cpp ourselves: it downloads the model, handles the GPU split, and it took minutes instead of an afternoon.
- think=false: Gemma reasons out loud by default, and that pass is the difference between 123 seconds an image and about 1.4 seconds. Leaving it on makes the demo impossible.
- temperature=0 with a fixed seed: at the default temperature the same photo gave three different lists in three runs. At zero, three runs came back byte identical.
- Gemma only fires at count events, not every frame: it takes about 4 seconds a call, and real surgical counts happen at defined moments anyway, before incision and before closure. The architecture matches the workflow.
- FastSAM instead of ordinary YOLO: YOLO trained on COCO only outlines the 80 things it was trained on, and FastSAM outlines anything, which is the point of a system meant to handle instruments it has never seen.
- FastSAM runs on CPU: every megabyte of VRAM belongs to Gemma.
- 512 pixel segmentation: 448 was noisy and 640 dropped the feed to about 4 fps, so 512 is the honest middle.
- At count-out we hand Gemma the count-in list and ask which items are gone, instead of running two inventories and diffing them: checking against a known list is a much easier question than producing the identical list twice.
- We count items rather than treat them as a set: two clamps go in and one comes out, and a set difference would report all clear while an instrument is still inside the patient.
- An unreadable reply from Gemma raises an error instead of returning an empty list: an empty list means the field is empty, which would flip every instrument to missing at once and put a red alarm on the projector because of a hiccup.
- We call 127.0.0.1 and never the word localhost: on Windows localhost tries IPv6 first, Ollama only listens on IPv4, and every single call was burning a fixed 2 seconds waiting for that to fail. It was 3.83 seconds an image before and 1.52 after, same model.
- keep_alive is set to never expire: Ollama unloads after 5 minutes by default, and the first count on stage would have cost 21 seconds instead of 4.
- The model gets warmed with one throwaway call at startup: a cold first call crashed llama-server once, and that must not happen on stage.
- There is a single-frame endpoint next to the video stream: a browser MJPEG image that drops once stays black forever, and polling single frames always recovers.
- A self-signed cert: the phone browser will not give up its camera on an insecure page, and there is no internet at the venue to get a real one.
- Everything is one Python file plus one Gemma module: two people and two agents are building this in a day, and the repo is the only shared memory.

### Considered and rejected

- gemma4:e4b-it-qat (6.1 GB, 7.5B, Q4_0). Tested Aug 1 around 12:45 PM on a real 1280x720 capture. It returned HTTP 500 after 13 seconds and llama-server died with exit status 0xc0000409, a stack buffer overrun, with a GGML_ASSERT failure. Reproducible, not a cold-start fluke. It does not fit on a 4 GB card. It is still pulled on disk, and it is not a fallback. E2B is the only working vision model on this machine.
- Cloud inference of any kind. There is a Cerebras account with gemma-4-31b on it and we did not use it. Using it would disqualify us from the On-Device Private Health track, and OR imagery is exactly the data that must not leave the building. Zero network calls is the pitch.
- gemma3n:e2b and gemma3n:e4b. Different model family (Gemma 3n) and text-only on Ollama. They cannot see an image. The names collide with our E2B and E4B, which makes it an easy mistake.
- gemma3:4b-it-q4_K_M. Pulled and sitting on disk, but Ollama lists its capabilities as completion only, with no vision.
- Fine-tuning YOLOv8 on instruments, which is what the 2023 prior art did. It rebuilds the old project, demotes Gemma to decoration, and there is no time in a one-day hackathon to collect and label a dataset.
- Renting a GPU to train a detector. Same reason. Training is the thing we designed the project around removing.
- COCO-trained YOLOv8n-seg as the segmenter. The weights are in the folder and unused. It only outlines the 80 classes it was trained on, which defeats the point.
- FastSAM-x (145 MB) and the other segmenters (mobile_sam, yoloe-11s-seg-pf) are all downloaded and unused. FastSAM-x is about 1.2 seconds a frame and produced no more usable masks than FastSAM-s.
- Video input to Gemma. Only the 31B supports it, and 31B is cloud-only for us.
- Running Gemma on every frame. About 4 seconds a call makes it impossible, and it has no memory between calls so it cannot hold object identity over time anyway.
- Two independent inventories diffed against each other. Even at temperature 0 it is fragile, because count-out is a different frame than count-in.
- Asking Gemma for a JSON list of everything in the zone during the autonomous loop. Measured live: it returned an empty array on a frame where the single-object prompt correctly answered 'phone'. Listing is the harder question and it was failing silently, so nothing ever reached the manifest. The loop names one object at a time instead.
- CUDA torch for the segmenter. Deliberately installed the CPU-only wheel so the GPU is entirely Gemma's.
- Detecting retained items inside the body. A retained sponge is retained precisely because it is hidden, and no camera can see it. We watch the instrument field, not the patient.
- TrialBridge, an earlier clinical trial matching idea with a voice agent. It needs the network for ClinicalTrials.gov and Twilio, which is incompatible with the on-device track.

### Hardware

- HP Victus 15-fb0xxx laptop. Ryzen 7 5800H, 16 GB system RAM, NVIDIA RTX 3050 Ti Laptop GPU with 4 GB VRAM, Windows 11. Roughly a $700 machine, which is part of the point.
- The bottleneck is the 4 GB of VRAM. Not the CPU, not the camera, not the network, because there is no network.
- Here is the measurement. Ollama reports the loaded model as 3.93 GB resident with only 1.46 GB of that in VRAM. So the majority of Gemma is running on the CPU, and the GPU is carrying somewhere around 37 percent of it. That partial offload is what sets the count latency, and it is why the bigger E4B model does not merely run slowly, it crashes.
- Measured latency from that: 3.7 to 4.5 seconds per count, mean 4.0 seconds over 3 runs on a real 1280x720 frame. Determinism held at that frame size, three runs byte identical.
- Resolution is effectively free on this box. A 1280 pixel image and a 256 pixel one both take about 3.7 seconds, which says the time is going to the model weights moving across the memory bus, not to the pixels.
- The camera is either the laptop webcam or an iPhone posting JPEG frames to the laptop over the USB cable, which is why the app serves HTTPS with a self-signed cert.
- Live frame rate of the segmentation overlay is computed in the app but I do not have a recorded number for it, so I am not going to quote one. What is recorded in the code is that 640 pixel inference dropped the feed to about 4 fps, which is why it runs at 512.
- One thing that is NOT measured: naming accuracy on a clean multi-object tray photo. The mechanism is verified (fast, deterministic, no hallucination on an empty scene) but accuracy has never been scored against ground truth. Do not claim an accuracy number on stage.

---

## Judge questions, by lens


### A clinician (nurse, surgeon, safety officer)

**[EASY] Be clear with me on one thing first. The count in my OR is two people, the circulator and the scrub tech, doing it out loud together and signing a count sheet. Does your system replace that, sit next to it, or something else? And if your screen says seven and my nurse says eight, whose number goes in the chart?**

> It does not replace it. The count is two people saying it out loud together, and we do not touch that. We watch the field and raise a hand when something we saw stops being there. If we disagree with the nurse, the nurse's number goes in the chart. We are a reason to recount, not a record.

Do not say: It automates the count so the nurse does not have to do it twice.

**[EASY] Physically, where does this camera live during a case? Because anything that ends up over my sterile field is a contamination path, and I am not adding a task to the one person in the room who already has too many.**

> Not in anyone's hands. It wants to be fixed above the back table or the Mayo stand, on a boom or the light head, pointed at the instruments and not at the patient. Today it is a phone I am holding, because that is what one build day buys you. Nothing about the method needs a person in the field.

Do not say: The circulating nurse can just hold it up when she needs a count.

**[MEDIUM] You have built an instrument counter. Somewhere around two thirds of retained items are sponges, not instruments. Why should the room care about the smaller half of my problem?**

> Sponges already have a product. RF tags and barcode counters are sold for sponges today. Nobody is putting a tag on every hemostat, so instruments are the part with no answer. And a blood soaked sponge on a bloody field is the hardest thing in the room to segment. We have not tried it and I am not going to claim it.

Do not say: Sponges would work exactly the same way, we just did not have any.

**[MEDIUM] A Kelly, a Crile and a mosquito are essentially the same instrument at three sizes. On my tray there are eight of them. Which of those did your model tell apart, and how do you know?**

> None. We have not put this in front of a real instrument set. We tested on ordinary objects standing in for tools. Three hemostats that differ mostly by size, with no scale reference in frame, is a case I expect it to fail. What is measured is speed and determinism. Naming accuracy on a tray is not measured, and I will not pretend otherwise.

Do not say: A vision language model should generalize to those fine.

**[MEDIUM] The retained items that frighten me most are fragments. A broken jaw tip, a screw, the end of a suction tip. Policy makes me inspect instruments for breakage for exactly that reason. Does your system see a fragment?**

> No. Our smallest tracked shape is about half a percent of the frame, and a broken jaw tip is smaller than that. Worse, a fragment breaks off inside the wound, so it was never on the field to be counted in. Nothing we built sees that. It is a real gap, not a threshold we can turn up our way out of.

Do not say: We could catch that by lowering the minimum area.

**[HOSTILE] Here is my problem with your core insight. My instrument count runs off a preprinted count sheet. Sterile processing assembled that tray and already knows exactly what is in it, down to the item. So what does zero shot naming actually buy me that the sheet does not?**

> The sheet says what should be in the tray. It does not say what is on the field, and things get opened onto the field off the sheet all case long. Hand me the tray list and our job gets easier, not pointless. Matching against a known list is a simpler question than naming cold. We just did not want to require the list to exist.

Do not say: Most hospitals do not really use count sheets in practice.

**[HOSTILE] In my hospital a count discrepancy is not a red banner on a screen. We stop, we call for an intraoperative film, the patient takes a dose, and the case runs long. So tell me what one false positive from your system costs me, and tell me your false positive rate.**

> That is the right way to score us. A false alarm costs a film, a dose and OR minutes. Our tracking flickers today when the segmenter drops a mask for a second or two, and I can show you that in the log. So this is a prompt to recount, not a trigger for an X-ray. At this false alarm rate you would not run it in a room yet.

Do not say: It is just a threshold we have not had time to tune.

**[HOSTILE] I read your code. Your manifest stops at eight items. A major lap tray is sixty or more instruments and I am counting all of them. What happens to instrument sixty one?**

> It is never on the manifest, so it can never be flagged. I want to say that plainly, because that is the dangerous direction. The cap is a demo guardrail so a dark frame cannot invent items, and it is sized for eight things on a table. We have not run this at tray scale. At tray scale the failure would be silence, not a false alarm.

Do not say: That is one constant, we can just raise it to sixty.

**[HOSTILE] Your screen says an instrument is inside the patient. You do not know that. It could be on the floor, in a drape fold, in the trash, or handed off to a runner. Why is your system making a claim it cannot support, on a screen, in front of a room?**

> You are right, and the label is wrong. All the geometry knows is that a shape it was tracking stopped being visible. It should read unaccounted for, which is the word the count actually uses. We will change it. Inside the patient is a claim we have no way to support, and our own scope rule says we do not make clinical determinations.

Do not say: It is just demo copy, everyone in the room understands what we mean.


### A senior ML engineer

**[EASY] In one sentence, what is Gemma actually responsible for, and what breaks if I delete it?**

> Gemma names each object once, the moment it enters the count zone, with no training data. That is the entire naming layer. Delete it and you have colored outlines with no names, and you are back to needing a labeled instrument dataset, which is exactly what the 2023 version had to build. It does not run the missing-item alarm. That part is geometry.

Do not say: Gemma powers the whole system. (A judge who reads the code will find that the alert never calls Gemma, and then everything else you said is suspect.)

**[HOSTILE] Open-vocabulary detectors like YOLO-World, OWL-ViT and GroundingDINO also name things zero-shot, with boxes, at frame rate. Why a 4.6B VLM instead?**

> Fair question, and we did not benchmark one. The difference is that those models need you to hand them the candidate vocabulary. You have to already know the word forceps before you can find forceps. Gemma produces the word. On a tray nobody has catalogued, that is the part we did not want to pre-specify. That comparison is the first thing I would run next.

Do not say: Those need training data too. They do not, and the judge knows it.

**[HOSTILE] Temperature zero with a fixed seed makes the same image reproducible. Your camera never sends the same image twice. What did you actually measure?**

> Exactly that, and no more. Three runs on one image, byte identical. Stability across different frames of the same object is not measured. That is why the design asks once, when the item arrives, and never asks again. The name is fixed for the life of that item, so there is no re-query to drift. Determinism buys us repeatability, not correctness.

Do not say: The model is deterministic, so naming is stable. Those are different claims.

**[HOSTILE] Your presence matcher accepts any blob within seventy percent of the stored area and a quarter of the frame diagonal, and the first match wins. Two similar instruments are interchangeable to that. What stops a neighbor's mask from covering a real disappearance?**

> Nothing does. First match wins, and the tolerances are wide because FastSAM's masks breathe frame to frame. Two similar tools sitting near each other can absolutely take each other's slot. In the demo it is eight items placed one at a time, so it holds. The honest fix is matching all items at once instead of greedily, and using the cutout appearance, not just area and centroid.

Do not say: The tolerances are tuned. They are wide because the segmenter is noisy, and saying tuned invites the follow-up you cannot answer.

**[HOSTILE] In a count system the only error that matters clinically is the item that never gets counted in. Your blocklist can reject it, the three second cooldown can swallow it, and the manifest caps at eight. All silently. What is your miss rate?**

> We do not have one. You have named the real hole. Cataloguing fails open and tracking fails safe, and the open direction is the dangerous one. What is on screen is the manifest count next to the live segmented count, so a mismatch is visible to the person running it. That is a cross-check, not a measurement, and I am not going to call it one.

Do not say: The blocklist handles that. It is fifty hand-written words, and it already let the word card through in testing.

**[MEDIUM] You claim you removed hand-built class knowledge. Then you hardcoded a fifty-word blocklist and a rule that rewrites Gemma's answer to fork whenever the outline is three times longer than it is wide. Isn't that just a smaller labeled prior?**

> The fork rule is fair to hit. It is a patch on one silhouette the model flips around on, and it should hint the display rather than overwrite the model. I would defend the blocklist differently. It lists what is never an instrument, hands, sleeves, the table. Adding a new instrument still takes zero work. That is the claim we are making, and that part survives.

Do not say: It is just a small heuristic. Own it as an override of the model, because that is what the code does.

**[MEDIUM] Everything is tuned on colorful household objects on a dark jacket. A real tray is specular steel on blue drape, shot top down, instruments overlapping. What breaks first?**

> Segmentation, before naming. The area and solidity filters and the overlap dedupe were tuned on separated objects with clean edges. Steel on steel, overlapping, merges into one mask, and one mask means one item to everything downstream. We have not measured any of it. That is a segmentation problem rather than an architecture problem, but I am not going to pretend it is small.

Do not say: Gemma has seen surgical instruments in training, so it will be fine. You have not tested it.

**[MEDIUM] You are running a Q4 quantization at roughly thirty seven percent GPU offload. How much naming accuracy did the quantization cost you?**

> We did not measure it. The larger E4B crashes on a four gigabyte card, so there is no bigger tier on this box to compare against, and we never ran an unquantized E2B. So the quantization is a hardware constraint we accepted, not a choice we validated. The one thing I will insist on is that any accuracy number we ever publish is measured on the quant we actually ship.

Do not say: QAT means there is no quality loss. QAT reduces the loss, it does not remove it, and you have no number either way.

**[MEDIUM] You have not run an accuracy evaluation. Design the one you skipped, right now.**

> Twenty photos of a real instrument tray. A person writes the ground truth names. Score top one name per instrument, and report the confusion pairs, because I expect the errors are clustered on a few similar silhouettes. Then the number that actually matters, how many instruments placed in the zone never produced a card at all. I would rather stand here with a number from twenty photos. Today we have none.

Do not say: We ran out of time. Say what you would measure and in what order, which shows you know what the number means.


### A healthcare product lead or investor

**[EASY] Hospitals already have RFID-tagged sponges and a mandatory verbal count. Why hasn't the incumbent solved this?**

> RFID tags sponges, not instruments. The tags do not survive repeated autoclaving and per-instrument cost is real, so the tagged systems cover soft goods and leave the metal to verbal counting. That gap is where we sit. I will say the honest thing too: soft goods are the bigger share of retained items, and we do not help there at all.

Do not say: RFID is expensive and outdated. That signals you never looked at what the incumbent actually covers.

**[MEDIUM] Is this a medical device? What is your FDA path, and do you think you clear the clinical decision support exemption?**

> Yes, it is a device. It informs a count decision at closure, and a nurse cannot independently review why a mask stopped matching, so I do not think it clears the CDS carve-out. Realistically it is a 510(k) or de novo against existing counting adjuncts, with a prospective study we do not have. Today is the technical claim, not the regulatory one.

Do not say: It is just decision support, so it is exempt. A judge who works in health will know that 'the clinician cannot review the basis' is exactly what kills that exemption.

**[HOSTILE] Your own README says naming accuracy is unmeasured. Would you put this in an operating room tomorrow?**

> No. Not tomorrow, and not without data. We measured latency and determinism. We did not measure whether the names are correct on a real instrument tray, and we tested on ordinary objects standing in for instruments. The claim I will defend is narrower: zero-shot naming removes the labeled dataset, and that part is real and reproducible.

Do not say: It worked great in our testing. There is no accuracy testing. Do not imply there is.

**[HOSTILE] The moment your system logs 'all accounted for', you have created a discoverable record. What happens to that log in a malpractice suit?**

> Honest answer, we had not thought that through, and you are right that it cuts both ways. Right now the log lives in memory and dies with the process, which is the wrong answer for audit and for liability. What it points at is a design rule: the system must never say all clear. It can raise a hand. It cannot sign off on a count.

Do not say: We would just add a database. The problem is not storage, it is what the system is allowed to assert.

**[HOSTILE] It false-alarms in the middle of closure. What does the OR actually do, and how do you avoid becoming another muted alarm?**

> Today it would flap. In our own event log an object that never moved flipped in and out repeatedly, because segmentation drops the mask for longer than our two second window. That is the first thing I would fix, because a flapping alarm gets muted and a muted alarm is worse than none. The rule has to be that a flag adds a check, never delays a closure on its own.

Do not say: False positives are safe because they fail toward caution. In an OR, a noisy alarm gets turned off, and then you have shipped nothing.

**[MEDIUM] Walk me through where this physically lives in an OR. Who holds the camera during a case?**

> Today it is a phone on a hotspot pointed at a table. In an OR it belongs fixed above the back table, where the scrub nurse lays instruments out, not over the wound. That scene is stable and well lit, which is the easy case for what we built. We have done nothing yet about blood, drapes, gloved hands crossing the view, or the overhead light.

Do not say: A nurse would just hold it up. Adding a task to the scrub nurse's hands in a sterile field is a non-starter and a clinician will say so.

**[HOSTILE] A seven hundred dollar gaming laptop cannot go in an operating room. Electrical safety, cleanability, uptime. Have you costed the real box?**

> Correct, and the laptop is a cost argument, not a deployment plan. What it proves is the compute envelope: four gigabytes of video memory and no network connection. That envelope fits on a medical grade box or an existing OR cart today. Electrical safety, cleanability and uptime are all real work, and none of it is started.

Do not say: Hospitals have laptops everywhere. Anything inside the OR has a different electrical and cleaning standard, and pretending otherwise reads as naive.

**[MEDIUM] Who signs the check? The OR director does not have a line item for this.**

> Not the OR budget. Risk management. A retained item is a never event, so the hospital eats the readmission, the second procedure and the settlement, and none of it is reimbursed. That is an insurance and risk line, which tolerates a per-room cost that capital equipment does not. We have no pricing and no pilot, so treat that as a thesis rather than a plan.

Do not say: We would sell it to hospitals as a SaaS subscription. Naming a channel you have not tested sounds like a pitch deck rather than a read of the buyer.

**[HOSTILE] Your whole pitch is that it names instruments nobody trained it on. From where I sit, an unbounded output space is the regulatory problem, not the feature. How do you validate a model that can say anything?**

> That is the sharpest version of the question. Live today, on a dark crop, the model answered 'a shirt' and 'card'. All that holds those out is a fifty word blocklist, which is thin. The architecture survives it because the name is a label for a human, and the alarm itself is geometry, which is bounded and testable. For a submission you freeze the vocabulary to one hospital's tray list and validate against that.

Do not say: Zero-shot means it generalizes, so it will be fine. That is the exact sentence a reviewer will use against you.


### A privacy and security specialist

**[EASY] Walk me through exactly what leaves this laptop during one count. Every call.**

> One call, to 127.0.0.1 on port 11434, where Ollama is running. That is loopback. It cannot route off the box. The web page loads no fonts, no CDN scripts, no remote images, so the browser makes no outside calls either. Pull the network adapter and the count still runs. We can do that here.

Do not say: It goes over an encrypted connection, so the images are safe.

**[EASY] What actually gets written to disk, and what happens to those files after this demo ends?**

> Two folders. static/catalog holds the thumbnail cards, and Reset deletes those files. captures/ holds a debug JPEG of the count zone every time we call the model, and nothing deletes those. They sit unencrypted on a Windows laptop. Both folders are gitignored so they never reach GitHub. Retention is a gap we have not closed, not a feature.

Do not say: Everything is wiped when you press Reset.

**[MEDIUM] You are serving HTTPS with a self-signed certificate. What does that certificate actually protect, and what does it not?**

> It does exactly two things. It encrypts the phone to laptop hop, and it makes Safari hand over the camera, which it will not do on an insecure page. It proves nothing about who the server is. The phone shows a warning you tap through. The private key sits unencrypted next to the code. In a hospital that would be a cert from their own internal CA.

Do not say: It is HTTPS, so the connection is secure.

**[HOSTILE] I am on your hotspot. What stops me from opening frame.jpg and watching the field, downloading your thumbnails, or posting my own frames into /ingest?**

> Nothing. There is no authentication on any route. Anyone on that link can pull the live frame and the thumbnail files, and yes, POST /ingest would let them feed a fake instrument field into the count. The only thing protecting it today is that the link is a cable between two devices we are holding. That is a demo condition, not a control.

Do not say: Nobody else would be on the network.

**[HOSTILE] You crop to the count zone before sending anything to the model, and you call that data minimization. But the zone is drawn over the patient. So the one image you hand to a language model is the most sensitive region in the frame.**

> That is a fair inversion. The crop keeps the room and the people standing in it out of every model call, which is real. But you are right that it aims at the field, not away from it. In an OR that rectangle sits over the drape and the tray, not a face, and a human drags it. Nothing in the code enforces that. It is a convention.

Do not say: The model only sees instruments.

**[HOSTILE] Your README says Absent makes no network calls of any kind. I can run a capture. Is that sentence true?**

> As written it is not. Ultralytics does a connectivity probe on import and posts anonymous usage events of its own. No image, no frame, but it is a call. We set YOLO_OFFLINE=1 and measured it at zero calls through import, model load and predict. The sentence should say no cloud inference and no data egress. That is the claim we can defend.

Do not say: That is a library doing it, not our code.

**[HOSTILE] Define on-device for me. Because I see a phone, a laptop, a network link and a separate server process.**

> It means no egress, not one chip. Two devices. The phone captures and sends JPEG frames, the laptop does every bit of inference, and the link is a USB cable with no route to the internet. Ollama is a second process on the same laptop reached over loopback. If you want the strict version, we run the identical count on the built-in webcam with the adapter disabled.

Do not say: It all runs on one device.

**[MEDIUM] Is this HIPAA compliant?**

> No, and we are not claiming it. No patient data was used at any point, we used household objects standing in for instruments. On-device removes the vendor and the business associate problem, which is the hardest part. Compliance also needs access control, an audit trail that survives a restart, and encryption at rest. We have none of those three yet.

Do not say: It is on-device, so HIPAA does not apply.

**[HOSTILE] Show me your audit trail. If a patient sues two years from now over a retained instrument, what record exists of what this system said?**

> None. The event log is a Python list in memory, capped at forty lines, and it dies with the process. Nothing is persisted. That is good for privacy and bad for everything else, because in a retained-item case that record is the evidence. A real version needs a signed append-only log with a stated retention period. Right now the privacy story and the legal story point opposite ways.

Do not say: The event log on screen is the record.


---

## What we would do with more time


### Next week

- Build a labeled evaluation set. This is the one that matters. Photograph a real instrument tray a hundred times, in different arrangements and different lighting, and hand-write the ground truth list for each photo. Then run count-in on all of them and score it. Right now the README says naming accuracy is unmeasured. In a week it does not have to be.
- Publish the number, whatever it turns out to be. Put an accuracy table in the README right next to the latency table we already measured. If Gemma names a hemostat correctly four times out of ten, that goes in the README. A bad number we measured is worth more than a good number we assumed.
- Score the count-out path separately. Take each photo, remove one known item, shoot it again, and measure two things: how often the system flags the right item, and how often it flags something that is still on the tray. Those are different failures and they need different numbers.
- Test overlap on purpose. Today, overlapping instruments being a weak point is our guess and not a measurement. Shoot the same tools clean, then touching, then stacked, and find where accuracy falls off. That tells us whether overlap is the main problem or a small one.
- Crop each segmented mask and name it one at a time, then score that against whole-frame naming on the same photo set. We already produce the masks, so this is a few hours of work. It is the obvious fix for overlap, but it costs seconds per count, so measure whether it actually buys accuracy before paying for it.
- Constrain the vocabulary. Give Gemma the list of instrument names that belong on the tray and let it pick from that list instead of writing free text. Our fuzzy name matcher exists because names drift between calls. A closed list removes the reason for the matcher.
- Make the repo runnable by someone else. requirements.txt, the segmentation weights or a script that fetches them, a LICENSE file, and the right port in the run command. Nobody can check our work on a repo they cannot start.

### Next six months

- Get real instruments in front of the camera, in sterile processing rather than the operating room. Trays get laid out and photographed there already, there is no patient in the room, and it is where counting actually begins. That is the low-risk way to get real data without touching a live case.
- Count against a known manifest instead of an open question. Real ORs use standardized trays with a printed count sheet. Checking a closed list of expected items is a much easier problem than asking a model what it sees, and it is a problem you can actually score.
- Fix the camera. A hand-held phone was right for a hackathon and wrong for everything after it. A fixed mount with known geometry, and probably a second angle, so occlusion can be resolved instead of guessed at.
- Stop letting the hardware pick the model. E2B is what runs on a 4 GB card, not what we would choose. On a proper edge box, run the larger variants and measure whether the accuracy is worth the cost and the latency. If it is not, that is a real finding.
- Track instruments through the whole case instead of at two moments. Log every time an item leaves the field and comes back. At closing, the system should have a history to reason over, not a single photograph.
- Design the human in, not around. The system proposes, the nurse confirms or corrects, and every correction is logged. That log is the audit trail and it is also the dataset that makes the next version better.
- Run a silent trial. Months of the system running next to a nurse who is counting normally, changing nothing, recording every disagreement. Nobody should act on this output until we know how often it is wrong and in which direction.
- Measure against the human baseline, not against zero. Manual counting has a known failure rate. The question is not whether we are accurate, it is whether we catch errors that people miss, and whether we add false alarms faster than we remove real misses.
- Harden the privacy claim into something auditable. On hospital-owned hardware, no network interface by construction rather than by policy, signed builds, and a written data flow a security review can read.

### What actually stands between this and an operating room

- Regulatory. Software that influences a surgical count is very likely a regulated medical device in the United States. That means an FDA pathway, a quality system, and clinical evidence. It is measured in years and money, not in weekends.
- Liability. If the screen says all clear and an item is left in a patient, someone has to own that. No hospital adopts this until that question is answered on paper, and the answer shapes the product. It probably has to stay advisory forever.
- The sterile field. Nothing unsterile goes over or near the field. The camera has to be draped or mounted outside the field entirely, which removes most of the good angles before we start.
- Camera placement. The view you want is straight down over the back table, and that space is already taken by surgical lights, arms, and people who are moving. Anything that blocks a surgeon's reach or the scrub tech's workspace gets removed on day one.
- Lighting. Surgical lights are extremely bright, very directional, and repositioned constantly. Stainless steel under a focused light is close to the worst case for a camera. Every photo we take on a desk is optimistic.
- Instruments in use do not look like instruments in a catalog. They are wet, they are covered in blood, they are wrapped in a towel or sitting in a nurse's hand. Recognition trained or tested on clean tools does not transfer for free.
- Occlusion is the normal state, not the edge case. Tools sit in piles. They get handed off, dropped in a basin, and covered by drapes. Our current approach sends one whole frame and already struggles when two objects touch.
- A retained sponge is invisible to any camera. It is retained precisely because it is inside the body. Sponges are the most common retained item, and this approach cannot see them at all. RFID-tagged sponges already exist and are the right answer for that half of the problem. We are not it.
- Alarm fatigue. A count assistant that raises false alarms gets ignored, then muted, then unplugged. The false-positive rate is more likely to kill this than the miss rate.
- The accuracy bar is unforgiving. A count system that is usually right is worse than no system, because it produces confidence that is not earned. We do not yet know what our accuracy is at all.
- Getting validation data legally. Real OR footage requires IRB approval and consent, and we cannot build the dataset we actually need without going through that process.
- Hospital procurement and IT. Biomed certification, security review, hardware lifecycle, and a purchasing cycle measured in quarters. Even a perfect system waits in that line.

### The honest version

What we built is a working mechanism, not a validated product. We showed that a small model running locally on a consumer laptop, with the wifi off, can name objects it was never trained on, fast enough and repeatably enough to run a count. That part is real and we measured it. What we did not measure is whether the names are correct, because we tested on ordinary objects on a table and never scored them against ground truth. So the first week is unglamorous. Build a labeled tray dataset and publish the accuracy number, whatever it says. If it is bad, we will know within a week whether the fix is cropping each object, constraining the vocabulary, or a bigger model. The six month version is not a better demo, it is sterile processing, where trays are already photographed and no patient is in the room, with real instruments, a real count sheet, and a silent trial running next to a nurse who is doing the count anyway. And here is the thing we will not claim: a retained sponge is invisible to any camera, and sponges are the most common retained item. We watch the instrument field, not the patient. Anyone who tells you a camera solves the whole problem is selling you something.
