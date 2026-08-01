# Absent

**An offline surgical equipment tracker. Nothing left behind.**

Track: On-Device Private Health

---

## The Problem

Surgeons leave items inside patients around 6,000 times a year, costing about $2.4 billion
in follow-up removal procedures.

Prevention today is a manual count: two people reciting numbers aloud in one of the most
distracting rooms in healthcare. It runs on human memory, and it fails.

A camera should fix this. Two things stop it. Detectors only recognize categories they were
trained on, and no hospital has a labeled dataset of its own trays. And OR video is too
sensitive to send to a cloud API.

---

## What Absent Does

A camera watches the instrument field. As each instrument enters, Absent names it zero-shot
and adds a manifest card with a photo of the actual object, then tracks it geometrically.
When a tracked item stops being visible, it is flagged as unaccounted for.

Nobody presses anything. The system detects motion, waits for the scene to settle, and
queries the model only when something changes.

---

## Why Gemma 4 is the Whole Project

Existing instrument trackers use fine-tuned detectors like YOLOv8, which means collecting
and hand-labeling a surgical instrument dataset first.

Gemma 4 is a vision-language model, so it names objects zero-shot. No dataset, no labeling,
no training run, and it generalizes to tools nobody trained a detector on. Remove Gemma and
this reverts to needing that dataset.

Responsibilities are split deliberately. Gemma names each object **once**, on entry. The
missing-item alarm never calls Gemma. That is spatial geometry running every frame, which is
why alerts are instant instead of costing an inference.

---

## Architecture

```text
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
  think=false, temperature=0, seed=42         ~1.5 s
        |
        v
  name -> blocklist -> fuzzy dedupe -> manifest card + cutout photo
        |
        v
  geometry tracker, every frame, no model
  object gone for >2 s  ------------> UNACCOUNTED FOR
```

**FastSAM** segments class-agnostically, so it outlines instruments nobody trained it on,
and runs continuously for the live feed. It cannot name anything.
**Gemma 4** names things, but holds no memory between calls, so it cannot track identity.
Neither can do the other's job.

---

## Stack

| Layer | Component | Reason |
|---|---|---|
| Naming | Gemma 4 E2B (`gemma4:e2b-it-qat`, Q4 QAT, 4.3 GB) | Fits 4 GB VRAM |
| Inference | Ollama 0.32.5, local HTTP | Loopback only, `keep_alive: -1` pins the model |
| Segmentation | Ultralytics FastSAM-s, `imgsz=512` | Class-agnostic |
| Tensors | PyTorch, CPU-only | Keeps all VRAM free for Gemma |
| Server | FastAPI + uvicorn, HTTPS | Safari requires HTTPS for camera access |
| Vision | OpenCV | Capture, JPEG, masks, morphology |
| Front end | Plain HTML and JS | Zero external requests, so offline is real |

---

## Running Fully Offline

Everything runs on one laptop (Ryzen 7 5800H, 16 GB RAM, RTX 3050 Ti with 4 GB VRAM).
Nothing is transmitted externally.

We verified rather than assumed. Instrumenting `socket.getaddrinfo`, `socket.connect`, and
`urllib.request.urlopen` showed the only outbound requests go to `127.0.0.1:11434`, loopback.
The interface has zero external dependencies, CDNs, or web fonts.

One finding: Ultralytics sends telemetry by default. `YOLO_OFFLINE=1` takes outbound calls
to zero.

---

## Benchmarks

| Metric | Result |
|---|---|
| Gemma naming latency | 1.4 to 1.6 s per call |
| Segmentation throughput | 7.4 fps at 1280x720 |
| Determinism | Byte-identical across runs (`temperature=0, seed=42`) |
| Empty scene | Returns an empty list, zero phantom objects |
| Disappearance to alert | ~2.0 s |

**Scope limitation.** Testing used surrogate items standing in for instruments.
Distinguishing visually identical instruments that differ only by scale, with no reference
in frame, is future work.

---

## Challenges and Mitigations

- **Prompt sensitivity.** The word "tray" made the model return empty predictions whenever
  no literal tray was visible. Removing it fixed the false negatives.
- **Reasoning overhead.** Default thinking cost ~123 s per image. `think=false` brought it
  to ~1.4 s.
- **Non-determinism.** Default sampling produced different names across runs, triggering
  false alerts. `temperature=0, seed=42` resolved it.
- **Pluralization.** "scissor" versus "scissors" flagged one item as present and missing at
  once. Fixed with normalization, substring, and fuzzy matching.
- **Multiset tracking.** Set differences missed duplicates: two clamps in, one out, reported
  all clear. Now tracked as explicit multisets.
- **Malformed responses.** An unparseable reply used to flag every item as missing. The
  parser now raises instead of defaulting.

---

## Scope and Safety

Decision support only. Absent assists the manual count. It does not replace clinical
personnel or make medical determinations. It reports that a tracked object is no longer
visible, and a human decides what that means. No patient data or real operating room video
was used at any point.

---

## Next Steps

Benchmark top-1 naming accuracy on real surgical trays under varied lighting and occlusion.
Then measure the metric that matters clinically: false-negative entry rate, meaning an item
enters the field and never registers a manifest card.

---

## Sources

- Cima RR, et al. Incidence and characteristics of potential and actual retained foreign
  object events in surgical patients. *Journal of the American College of Surgeons*, 2008.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/18589366/)
- [Gemma 4](https://ai.google.dev/gemma), Google DeepMind, run locally via
  [Ollama](https://ollama.com).
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) via
  [Ultralytics](https://github.com/ultralytics/ultralytics).
