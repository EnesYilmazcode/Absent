# Absent

**An offline surgical equipment tracker. Nothing left behind.**

Track: On-Device Private Health

---

## The problem

Surgical teams count instruments before a procedure and again before closing. The count is
two people saying numbers out loud in the most distracting room in the hospital, and it
still fails. Retained surgical items are classified as a "never event", meaning they are
supposed to be impossible, and they happen a few thousand times a year in the US alone
(Cima et al., *J Am Coll Surg*, 2008).

The obvious fix is a camera that watches the instrument field. The reason nobody ships one
is that a detector only recognizes classes somebody trained it on, and no hospital has a
labeled photo set of its own trays.

The other reason is privacy. Operating room imagery is about the most sensitive data that
exists, and a system that streams it to a cloud API is a system no hospital will install.

## What Absent does

A camera watches the instrument field. As each instrument enters the field, Absent names it
and adds a card to a manifest with a photo of the actual object. Then it tracks that object
continuously. If something on the manifest stops being visible, it is flagged as
unaccounted for.

Nobody presses a button. The system watches for motion, waits for the scene to settle, and
asks only when something has changed.

## Why Gemma 4 is the whole project

The prior art here is Uncountable, which won HackOHI/O 2023 doing real-time instrument
tracking. It needed a **fine-tuned YOLOv8**, and the team spent their hackathon collecting
and hand-labeling a dataset of surgical instruments, because that is what a detector
requires.

Gemma 4 is a vision-language model, so it names objects zero-shot. No dataset, no labeling,
no training run, and it generalizes to instruments nobody ever trained a detector on.

Delete Gemma and this project reverts to needing a labeled dataset it does not have. That
is the test for whether a model is load-bearing, and Gemma passes it.

We are specific about what Gemma does and does not do. Gemma names each object once, when
it enters the count zone. The missing-item alarm is not Gemma. That is geometry, running
every frame, which is why the alert is instant rather than costing an inference.

## Architecture

```
  iPhone (Safari, getUserMedia)
        |  JPEG over HTTPS, self-signed cert, USB tether or hotspot
        v
  POST /ingest  ------------------> decode to BGR ndarray
        |
        v
  capture thread, 7.4 fps
        |
        +--> FastSAM-s (CPU) ------> class-agnostic masks
        |                            filtered by area, solidity,
        |                            border contact, IoU dedupe
        |
        +--> motion (96x96 absdiff) -> "has the scene changed?"
        |
        v
  watch thread, samples every 100 ms
        |
        |  scene stirred, then held still ~0.8 s
        |  mask count went UP, and the shape is not already catalogued
        v
  crop count zone -> upscale to >=768 px -> JPEG -> base64
        |
        v
  Gemma 4 E2B via Ollama, 127.0.0.1:11434
  think=false, temperature=0, seed=42          ~1.5 s
        |
        v
  name -> blocklist -> fuzzy dedupe -> manifest card + cutout photo
        |
        v
  geometry tracker, every frame, no model
  object gone for >2 s  ------------> UNACCOUNTED FOR
```

Two models with two different jobs. FastSAM is class-agnostic, so unlike a COCO-trained
detector it outlines instruments nobody trained it on, and it runs continuously so the feed
is live. It cannot name anything. Gemma names things but has no memory between calls, so it
cannot hold object identity over time. Neither model can do the other's job.

## Stack

| Layer | What we used | Why |
|---|---|---|
| Naming | Gemma 4 E2B, `gemma4:e2b-it-qat` (4.3 GB, Q4 QAT) | Fits 4 GB of VRAM. E4B crashes on this card |
| Inference | Ollama 0.32.5, local HTTP | Loopback only, `keep_alive: -1` pins the model |
| Segmentation | Ultralytics FastSAM-s, `imgsz=512` | Class-agnostic. 640 dropped the feed to 4 fps |
| Tensors | PyTorch 2.13 **CPU-only** | Deliberate. All VRAM stays free for Gemma |
| Server | FastAPI + uvicorn, HTTPS | Safari refuses camera access over plain HTTP |
| Vision utils | OpenCV | Capture, JPEG, masks, morphology |
| Front end | Plain HTML and JS, no framework | Zero external requests, so offline is real |

## Running offline

Everything runs on one laptop: a Ryzen 7 5800H with 16 GB of RAM and an RTX 3050 Ti with
4 GB of VRAM. Nothing is streamed anywhere.

We checked this rather than assuming it. We instrumented `socket.getaddrinfo`,
`socket.connect` and `urllib.request.urlopen`, then imported the whole app and ran
inference. The only call our code makes goes to `127.0.0.1:11434`, a literal loopback
address. The HTML has zero external references, no CDN and no web fonts, so the interface
works with the network adapter disabled.

One honest finding from that audit: Ultralytics itself sends usage analytics by default.
`YOLO_OFFLINE=1` takes the measured outbound call count to zero. We would not have found
that by reading our own code.

## What we measured, and what we did not

Measured on this machine, on real captures:

| | Result |
|---|---|
| Gemma per naming call | 1.4 to 1.6 s |
| Live segmentation | 7.4 fps at 1280x720 |
| Determinism | 3 runs on one frame, byte identical at `temperature=0, seed=42` |
| Empty scene | Returns an empty list. It does not invent objects |
| Time from an item disappearing to the alert | about 2 s |

Not measured: **naming accuracy on a real instrument tray.** We tested on ordinary objects
standing in for instruments, per the synthetic-data rule. Three hemostats that differ mostly
by size, with no scale reference in frame, is a case we expect to fail. We would rather say
that than publish a number we did not earn.

## What went wrong, and what we did about it

- **Asking about a "tray" returned nothing.** Gemma answered with an empty list whenever it
  could not see a literal surgical tray, which was every frame. Removing the word fixed it.
- **Thinking mode cost 123 seconds per image.** `think=false` brought it to 1.4 s.
- **At default temperature, three runs gave three different lists.** Set differences over
  names that unstable invent a missing instrument on almost every count. Temperature 0 with
  a fixed seed made it repeatable.
- **"scissors" versus "scissor" reported an item as present and missing at once.** We match
  answers back onto the manifest with plural, substring, then fuzzy matching.
- **Set membership hid a real failure.** Two clamps in, one out, and a set difference says
  all clear while an instrument is still unaccounted for. We count with a multiset instead.
- **An unreadable reply used to flag everything.** A single malformed response reported
  every instrument as missing. It now raises rather than falling back.

## Attribution

Absent is built on top of [Uncountable](https://github.com/gulkoa/uncountable) by Alex Gulko
and David Novikov, which won 1st place at HackOHI/O 2023 doing real-time surgical instrument
tracking with a fine-tuned YOLOv8. It is AGPL-3.0.

We say this plainly because it forces the useful question, which is what Gemma added. The
answer is clean: it removed the training requirement entirely. Their system needed a labeled
instrument dataset before it could recognize a single tool. This one needs none.

## Scope and safety

Decision support only. Absent assists the count. It does not replace the nurse, and it does
not make a clinical determination. It says an object it was tracking is no longer visible,
and a human decides what that means. No diagnosis, no treatment recommendation. All testing
used ordinary objects standing in for instruments. No real patient data and no operating
room footage were used at any point.

## Next

The first thing is the number we do not have. Twenty photos of a real instrument set, human
ground truth, top-1 name per instrument, and the confusion pairs, because we expect the
errors cluster on a few similar silhouettes. Then the metric that actually matters
clinically: how often an instrument placed in the field never produced a card at all, since
that is the failure that fails open.
