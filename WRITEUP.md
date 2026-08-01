# Absent

**An offline surgical equipment tracker. Nothing left behind.**

Track: On-Device Private Health

---

## The Problem

Surgeons leave items inside patients around 6,000 times a year, costing about $2.4 billion
to perform follow-up removal procedures.

The standard prevention method today is a manual count. Two team members recite numbers
aloud before the procedure and again prior to closing, in one of the most distracting
environments in healthcare. It relies entirely on human memory, and it fails.

The intuitive fix is an automated camera system monitoring the instrument field. The reason
this isn't widely deployed is that traditional object detectors only recognize categories
they were explicitly trained on, and hospitals do not possess labeled photo datasets of
their specific instrument trays.

The second barrier is privacy. Operating room imagery is highly sensitive data. A system
streaming live OR video to a cloud API is a system hospitals will not install.

---

## What Absent Does

A camera continuously monitors the instrument field. As each instrument enters the field,
Absent names it zero-shot and adds a card to a visual manifest alongside a photograph of the
actual object. It then tracks that object's spatial geometry continuously. If a tracked item
on the manifest stops being visible, it is immediately flagged as unaccounted for.

No manual user interaction is required. The system detects motion, waits for the scene to
settle, and queries the visual model only when an active change occurs.

---

## Why Gemma 4 is the Whole Project

Traditional real-time surgical instrument tracking systems rely on fine-tuned detection
models (like YOLOv8), requiring teams to spend extensive time collecting and manually
labeling surgical instrument datasets.

Gemma 4 is a vision-language model (VLM), enabling zero-shot object naming. It requires no
dataset creation, no manual labeling, and no training runs, while generalizing effectively
to unique tools and instruments without prior training.

Without Gemma 4, this architecture reverts to requiring a labor-intensive, labeled dataset.

The responsibilities are strictly separated. Gemma names each object once upon entering the
count zone. The missing-item alarm does not depend on Gemma. It relies on low-overhead
spatial geometry running every frame, making alerts instantaneous without incurring
continuous inference costs.

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

Two distinct models fulfill distinct roles.

**FastSAM** handles class-agnostic segmentation. It outlines arbitrary instruments without
prior category training and runs continuously to maintain a live feed. It cannot generate
text names.

**Gemma 4** generates object names zero-shot. It lacks state memory between distinct calls,
so it cannot maintain long-term object identities over time.

---

## Stack

| Layer | Component | Reason |
|---|---|---|
| Naming | Gemma 4 E2B (`gemma4:e2b-it-qat`, 4.3 GB, Q4 QAT) | Fits within 4 GB VRAM budgets |
| Inference | Ollama 0.32.5, local HTTP | Loopback-only communication (`keep_alive: -1` pins the model) |
| Segmentation | Ultralytics FastSAM-s (`imgsz=512`) | Class-agnostic processing |
| Tensors | PyTorch CPU-only | Offloads tensor work so VRAM stays reserved for Gemma |
| Server | FastAPI + uvicorn, HTTPS | HTTPS required for Safari camera access |
| Vision Utils | OpenCV | Frame capture, JPEG encoding, mask handling, morphology |
| Front End | Plain HTML and JavaScript | Zero external web requests, guaranteeing offline execution |

---

## Running Fully Offline

The system runs entirely on a single edge machine, tested on a Ryzen 7 5800H with 16 GB RAM
and an RTX 3050 Ti with 4 GB VRAM. No data is transmitted externally.

To verify offline isolation, we instrumented `socket.getaddrinfo`, `socket.connect`, and
`urllib.request.urlopen` across the runtime environment. The only outbound network requests
produced connect directly to `127.0.0.1:11434` (loopback). The web interface includes zero
external dependencies, CDNs, or web fonts.

Note: Ultralytics sends usage telemetry by default. Setting `YOLO_OFFLINE=1` eliminates all
external network calls entirely.

---

## Performance and Benchmarks

| Metric | Result |
|---|---|
| Gemma naming latency | 1.4 s to 1.6 s per call |
| Live segmentation throughput | 7.4 fps at 1280x720 |
| Determinism | Byte-identical across runs (`temperature=0, seed=42`) |
| Empty scene handling | Returns empty lists, zero invented phantom objects |
| Disappearance to alert latency | ~2.0 seconds |

**Scope limitation.** Testing utilized standardized surrogate items representing medical
instruments. Fine-grained classification between visually identical instruments of differing
scale without reference points remains an area for future evaluation.

---

## Key Technical Challenges and Mitigations

**Prompt sensitivity.** Using generic terms like "tray" caused the VLM to return empty
predictions if a literal tray was absent. Refining prompt constraints resolved zero-detection
false negatives.

**Reasoning overhead.** Default reasoning modes added ~123 seconds of processing per image.
Explicitly disabling thinking (`think=false`) reduced execution time to ~1.4 seconds.

**Non-deterministic outputs.** Default sampling generated variant naming tokens across runs,
triggering false missing-item alerts. Enforcing deterministic parameters (`temperature=0,
seed=42`) resolved label variance.

**String matching and pluralization.** Differences between singular and plural outputs (for
example "scissor" versus "scissors") resulted in simultaneous false positive and false
negative state flags. Matching logic was updated to use string normalization, substring
checks, and fuzzy string distance metrics.

**Multiset tracking.** Basic set differences failed when handling duplicate object classes,
such as multiple identical clamps. The tracking state was updated to operate on explicit
multisets.

**Malformed response handling.** Unparseable model responses previously defaulted to
flagging all active items as missing. The parser was updated to raise soft retry exceptions
instead of defaulting state changes.

---

## Scope and Safety

**Decision support only.** Absent is designed to assist manual count protocols. It does not
replace clinical personnel or issue autonomous medical determinations. It highlights when a
tracked object is no longer detected in the visual frame for human review. No patient data
or live operating room video was used during development or testing.

---

## Next Steps

Benchmarking top-1 naming accuracy against curated datasets of real surgical trays under
varied lighting and occlusion conditions.

Measuring clinically critical edge metrics, specifically false-negative entry rates, meaning
cases where an item enters the field but fails to register a manifest card.

---

## Sources

- Cima RR, et al. Incidence and characteristics of potential and actual retained foreign
  object events in surgical patients. *Journal of the American College of Surgeons*, 2008.
  [PubMed entry](https://pubmed.ncbi.nlm.nih.gov/18589366/)
- [Uncountable](https://github.com/gulkoa/uncountable), Alex Gulko and David Novikov, 1st
  place HackOHI/O 2023. The fine-tuned YOLOv8 approach this project builds on. AGPL-3.0.
- [Gemma 4](https://ai.google.dev/gemma), Google DeepMind. Run locally via
  [Ollama](https://ollama.com).
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) via
  [Ultralytics](https://github.com/ultralytics/ultralytics).
