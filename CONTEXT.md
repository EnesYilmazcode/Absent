# CONTEXT.md

**Living document. Update it whenever a decision changes. Anyone joining this repo,
human or agent, reads this file first.**

Last updated: Aug 1, 2026 — build day.

---

## What we're building

# Absent
### Nothing left behind.

Repo: https://github.com/EnesYilmazcode/Absent

Surgical instrument count verification. A camera watches the instrument field.
At "count in" and "count out" the system inventories what it sees. Anything present
at count-in but missing at count-out gets flagged as unaccounted for.

**Why it matters:** retained surgical items are a "never event" — roughly 1 in
5,500–7,000 operations. The existing mitigation is manual verbal counting, which
fails under distraction.

**Why it must be on-device:** OR imagery is among the most sensitive data that
exists. Nothing can be streamed to a cloud API. Everything runs locally.

---

## The core insight (this is the whole project)

The conventional way to build this needs a **fine-tuned YOLOv8** — collect and label
a custom dataset of instruments first, because YOLO only recognizes classes it was
trained on.

**We replace the fine-tune with Gemma.** Gemma 4 is a vision-language model, so it
names objects zero-shot with no training data at all. That means:

- No dataset collection, no labeling, no training run
- Generalizes to instruments nobody ever trained a detector on
- Gemma is genuinely load-bearing, not decorative

That sentence is the pitch and the Gemma Integration score.

---

## Architecture

```
webcam ──> YOLOv8 (class-agnostic, pretrained, CPU)  ──> boxes, live, ~30fps
                          │
                    [count event]
                          ▼
                    crop each box
                          ▼
              Gemma 4 E4B via Ollama (local, GPU)  ──> object names (JSON)
                          ▼
              set difference: count_in vs count_out
                          ▼
                    flag unaccounted items
```

**Simpler path to try first:** skip YOLO entirely. Send the whole frame to Gemma and
ask for a JSON inventory. One call instead of a detection pipeline plus N crops. Add
YOLO only if whole-image inventory misses small or overlapping objects.

### Why Gemma only fires at count events, not every frame

E4B on an RTX 3050 Ti is seconds per image, so continuous per-frame inference is
impossible. It also has no memory across calls, so it cannot maintain object identity
over time — that is YOLO + tracker's job.

This is not a compromise. **Real surgical counts happen at defined moments**
(before incision, before closure), not continuously. The architecture matches the
actual clinical workflow.

The demo still reads as fully live because the YOLO overlay runs continuously on the
video feed the whole time.

---

## Hardware and stack

- **Machine:** HP Victus 15-fb0xxx — Ryzen 7 5800H, 16 GB RAM, RTX 3050 Ti (4 GB
  VRAM), Windows 11. Disk was nearly full; cleared on build day.
- **Model:** Gemma 4 **E2B, QAT quant** via Ollama — `gemma4:e2b-it-qat`.
  - Verified sizes on the Ollama library (Aug 1, the earlier "~3 GB at Q4" note was
    wrong): `e2b-it-qat` **4.3 GB**, `e4b-it-qat` **6.1 GB**, `e4b` (default q4_K_M)
    **9.6 GB**, `12b-it-qat` 7.2 GB.
  - E2B and E4B both list **Supported Modalities: Text, Image, Audio** — vision is
    confirmed on the small variants, we do not need a 12B.
  - Starting on E2B because 4 GB VRAM means anything over ~4 GB spills to system RAM
    and gets slower.
  - **`e4b-it-qat` is NOT a fallback. It is broken on this machine.** Tested Aug 1
    ~12:45 PM on a real 1280x720 capture: HTTP 500 after 13 s, `llama-server process
    has terminated: exit status 0xc0000409` (stack-based buffer overrun) with a
    `GGML_ASSERT` failure. Reproducible, not a cold-start fluke. **E2B is the only
    working vision model on this box. There is no quality fallback. Do not burn demo
    time trying to switch models.**
- **Do NOT use `gemma3n:e2b/e4b`** — different family (Gemma 3n) and **text-only on
  Ollama**. It cannot see images. Name collision with our E2B/E4B, easy mistake.
- **Endpoint:** `http://localhost:11434/v1` (OpenAI-compatible), model
  `gemma4:e2b-it-qat`
- **Detection:** Ultralytics YOLOv8n, **CPU-only torch** — keep all VRAM for Gemma
  - `pip install ultralytics --index-url https://download.pytorch.org/whl/cpu`
- **No cloud.** Cerebras (`gemma-4-31b`) exists as an account but is NOT used —
  using it would disqualify us from the On-Device track.

---

## The go/no-go test

**Nothing else matters until this passes.** Put 8 visually distinct objects on a
table, good lighting, no overlap. Photograph. Then run twice on the same image:

```powershell
ollama run gemma4:e2b-it-qat "List every object as a JSON array of names. Be exact, no extras." ./tray.jpg
```

- Both runs return the same clean list → build it
- Hallucinated objects or the two lists disagree → the naming layer is unreliable,
  and no amount of YOLO fixes that. Fall back.

Record the result of this test here when it's run.

**Result (Aug 1, ~12:15 PM): mechanism PASSES. Naming accuracy on a clean tray photo
still untested** because the only instrument images on hand were dark, top-down and
low contrast.

Run on one of those reference instrument photos:

| Setting | Time per image | Consistency |
|---|---|---|
| Defaults (thinking on) | **123 s** | crashed once, then 1 usable answer |
| `think=false` | **1.2–1.5 s** | 3 runs, 3 *different* lists |
| `think=false`, `temperature=0`, `seed=42` | **1.4 s** | 3 runs, **byte-identical** |

### Two settings are mandatory, not optional

1. **`think=false`.** Gemma 4 does a visible reasoning pass by default. That pass is
   the entire difference between 123 s and 1.4 s per image. Leaving it on makes the
   demo impossible.
2. **`temperature=0` plus a fixed `seed`.** At default temperature the same image
   gave `["paper","scissors","piece_of_paper"]`, then
   `["paper","scissors","clipboard","clipping tool","scissors"]`, then
   `["paper","scissors","clip","wooden floor"]`. Set difference over names that
   unstable would invent missing instruments on every count. At temperature 0 the
   three runs were identical.

### Consequence for the architecture

Do **not** run two independent inventories and diff them. Even at temperature 0 that
is fragile, because count-out sees a different frame than count-in.

Instead: at count-out, **pass the count-in list into the prompt** and ask Gemma which
of those named items are no longer visible. Comparison against a known list is a much
easier task than re-deriving an identical list from scratch.

### Measured runtime facts

- **Latency is ~4 s per count, not 1.4 s.** The 1.4 s number was measured on a small
  dataset image. On the frames the app actually sends (1280x720 webcam capture) E2B
  takes **3.7–4.5 s, mean 4.0 s** over 3 runs. Still fine for a count event, but budget
  4 s of dead UI per button press when rehearsing, and consider downscaling the frame
  before encoding if it needs to be faster.
- **Determinism holds at the real frame size.** Same 3 runs returned byte-identical
  output at `temperature=0, seed=42, think=false`.
- **No hallucination on an empty scene.** Given a webcam frame with no tray in it, E2B
  returned `[]` under the inventory prompt. It does not invent instruments.
- E2B wraps its JSON in a ```` ```json ```` fence. `gemma._parse` already strips this.
- The inventory prompt says to ignore the background and E2B still emitted
  `"background"` on a non-tray frame. Worth one prompt tightening pass if the real tray
  run shows the same, but do not touch it until the tray run is done.
- Ollama **0.32.5**, model loads at **63% CPU / 37% GPU** on the RTX 3050 Ti (4 GB).
  Partial offload, and 1.4 s per image is still fine. Do not fight this.
- One crash seen on a cold first call: `llama-server` died with
  `CUDA error: shared object initialization failed`. It has not recurred. **Warm the
  model with one throwaway call before the demo** so a cold-start crash cannot happen
  on stage.
- Calling through `POST http://localhost:11434/api/generate` with base64 in `images`
  is what we tested and what works.

---

## Event constraints

| | |
|---|---|
| Event | Build with Gemma NYC: On-Device AI for Healthcare |
| Venue | Celonis, One World Trade Center, floor 70 |
| **Submissions lock** | **4:00 PM** (moved from 3:45 in the morning blast) |
| Presentations | 4:00 PM live |
| Winners | 5:30 PM |
| Track | **On-Device Private Health** |
| Prize pool | $2,000 across three tracks |
| Submission | Kaggle |

### Hard rules

- Gemma 4 must be **core**, not decorative
- **Decision support only.** No diagnosis, no treatment recommendations.
  Our framing: the system *assists* the count, it does not replace the nurse or
  make a clinical determination.
- **Synthetic or public data only.** No real patient data, no real OR footage.
  We use ordinary objects on a table standing in for instruments.

### Rubric (100 pts)

| Weight | Criterion | Our position |
|---|---|---|
| 30 | Healthcare Impact | Strong — never-event, well-documented problem |
| 25 | Gemma Integration | Strong — zero-shot naming replaces a fine-tune |
| 20 | Functionality | Depends entirely on the go/no-go test |
| 15 | Presentation & Writeup | Start the writeup by 2:30, do not leave to the end |
| 10 | Privacy & Safety | Strong — fully offline, wifi off during demo |

---

## Demo plan

1. Camera live on screen, YOLO boxes tracking objects as they move
2. **Turn wifi off in front of the room.** This is the whole pitch, made visible.
3. Hit "count in" — system inventories the field
4. **Hand the tray to a judge. Let them remove an object.** Unfakeable — no
   pre-recording survives judge-supplied input.
5. Hit "count out" — system names exactly what's missing
6. Close: this ran on a $700 laptop with no internet. OR imagery never left the
   machine.

**Insurance: screen-record the working demo the moment it works.** The rubric allows
"live demo **or video**." Record it regardless of how well things are going.

---

## Decided / rejected

**Decided:**
- On-Device Private Health track
- Gemma zero-shot naming instead of fine-tuning a detector
- Count-event inference, not per-frame
- E4B local via Ollama

**Rejected, and why:**
- *Fine-tuning YOLO on instruments* — rebuilds the conventional approach, demotes
  Gemma to decoration, and there is no time to collect and label a dataset
- *Renting a GPU to train* — same reason; training is the thing we designed around
- *Cerebras / any cloud inference* — disqualifies the On-Device track
- *Video input to Gemma* — only supported on 31B, which is cloud-only for us
- *Detecting items inside the body* — a retained sponge is retained because it's
  hidden. No camera can see it. We watch the instrument field, not the patient.
- *TrialBridge* (clinical trial matching + voice agent) — earlier direction, needs
  network for ClinicalTrials.gov and Twilio, incompatible with On-Device

---

## Open questions

- [x] Does the go/no-go naming test pass? **Mechanism yes** (1.4 s, deterministic).
      Accuracy on a clean 8-object tray photo is still unmeasured.
- [ ] Does class-agnostic YOLO give usable boxes without a fine-tune?

---

## Conventions for this repo

- **Update this file when a decision changes.** Stale context is worse than none.
- Commit often with real messages — agents read the log
- Keep secrets out of the repo (there shouldn't be any; we're fully local)
- Anything an agent should not touch goes in `.gitignore` and gets noted here
