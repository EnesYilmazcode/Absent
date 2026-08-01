# Absent

### Nothing left behind.

`on-device` · `no network` · `Gemma 4 E2B via Ollama` · `AGPL-3.0`

Surgical instrument count verification that runs entirely on one laptop. A camera
watches the instrument field. At "count in" the system names every object it sees.
At "count out" it checks that same list against the tray and flags anything it can
no longer account for.

---

## The problem

Retained surgical items are classified as a "never event", and they still happen in
roughly 1 in 5,500 to 7,000 operations. The standing mitigation is a nurse counting
instruments out loud twice, which is a memory-and-attention task performed in the
middle of the most distracting environment in the hospital.

## The core insight

The conventional way to build this needs a **fine-tuned YOLOv8**: collect and
hand-label a custom dataset of surgical instruments first, because a detector only
recognizes classes somebody trained it on.

Gemma 4 is a vision-language model, so it names objects zero-shot. That removes the
entire training requirement:

- No dataset collection, no labeling, no training run
- It generalizes to instruments nobody ever trained a detector on
- Delete Gemma and the project reverts to needing a labeled dataset it does not have

That is why Gemma is load-bearing here rather than decorative. It is not summarizing the
output of a real system. It is the part that knows what a thing is called.

## Why fully on-device

Operating room imagery is among the most sensitive data that exists. Absent makes no
network calls of any kind: the model runs under a local [Ollama](https://ollama.com)
server, the segmentation weights are a local file, and the web UI has zero external
resources (no CDN scripts, no web fonts, no remote images). The demo turns wifi off in
front of the room and keeps working, which is the privacy claim made visible rather than
asserted.

---

## Architecture

```
webcam ──> FastSAM-s, class-agnostic segmentation, CPU ──> live mask overlay
                          │
                    [count event]
                          ▼
                  full frame, JPEG, base64
                          ▼
    Gemma 4 E2B via Ollama (local, think=false, temperature=0, seed=42)
                          ▼
   count in: name every object        count out: check the count-in list
                          ▼
            anything not confirmed present ──> unaccounted for
```

Two models, two different jobs. FastSAM is class-agnostic, so unlike a COCO-trained
detector it outlines instruments nobody trained it on, and it runs continuously so the
feed reads as live. It cannot name anything. Gemma names things but has no memory across
calls, so it cannot hold object identity over time.

Note that `CONTEXT.md` still carries an older diagram with YOLOv8 and per-box crops. The
shipped pipeline is the one above: FastSAM for the continuous overlay, whole uncropped
frame to Gemma at count events. `yolov8n-seg.pt` and `mobile_sam.pt` sit in the folder
unused.

### Why Gemma fires only at count events

E2B takes about 1.4 s on a small image and 3.7 to 4.5 s on a full 1280x720 webcam frame,
so per-frame inference is not possible on this hardware. That constraint turned out to
match the clinical workflow rather than fight it. Real surgical counts happen at defined
moments, before incision and before closure, not continuously. Absent counts when a nurse
would count.

### The count-out design decision

Absent does **not** run two independent inventories and diff them. The count-in list is
passed into the count-out prompt, and Gemma is asked which of those named items are still
visible. Comparing against a known list is a much easier task than re-deriving an
identical list from scratch on a different frame with different lighting and occlusion.

Then the count-in list is trusted over the model. Anything Gemma fails to mention in
either bucket is appended to `missing` (`gemma.py`, `check_against`), so an omission
fails toward "unaccounted for" instead of quietly disappearing.

This is a consequence of measurement, not taste. At default temperature the same image
returned `["paper","scissors","piece_of_paper"]`, then
`["paper","scissors","clipboard","clipping tool","scissors"]`, then
`["paper","scissors","clip","wooden floor"]`. A set difference over names that unstable
would invent a missing instrument on almost every count.

---

## Measured, not claimed

All numbers below were measured on the machine listed under Hardware, on real capture
frames, through `POST http://localhost:11434/api/generate` with base64 in `images`.

| Setting | Time per image | Consistency across 3 runs |
|---|---|---|
| Defaults, thinking on | 123 s | crashed once, then 1 usable answer |
| `think=false` | 1.2 to 1.5 s | 3 runs, 3 different lists |
| `think=false`, `temperature=0`, `seed=42` | 1.4 s | 3 runs, byte-identical |

Both settings are mandatory. Gemma 4 does a visible reasoning pass by default, and that
pass is the whole difference between 123 s and 1.4 s. Temperature 0 with a fixed seed is
what makes the names stable enough to compare.

Other measurements:

- On the 1280x720 frames the app actually sends, E2B takes 3.7 to 4.5 s, mean 4.0 s over
  3 runs, and the output stayed byte-identical at `temperature=0, seed=42`.
- Given a webcam frame with no tray in it, E2B returned `[]`. It did not invent
  instruments on an empty scene.
- The model loads at roughly 63% CPU / 37% GPU on a 4 GB card. Partial offload, and the
  latencies above already include it.
- E2B wraps its JSON in a ```` ```json ```` fence. `gemma._parse` strips it.

**What is not measured: naming accuracy on a clean tray photo.** The results above say
the mechanism works and repeats. They do not say the names are correct. The only images
available during the build were reference instrument frames, which are dark, top-down
and low contrast. Treat accuracy as unverified.

---

## Setup

Windows, PowerShell, from the repo root.

**Requirements**

- Python 3.11+ (built on 3.13)
- [Ollama](https://ollama.com) 0.32.5 or newer
- `gemma4:e2b-it-qat` (4.3 GB)
- `fastapi`, `uvicorn`, `opencv-python`, `requests`, `ultralytics`, `numpy`
- `FastSAM-s.pt` (24 MB) in the repo root

**1. Pull the model**

```powershell
ollama pull gemma4:e2b-it-qat
```

> **Do not use `gemma3n:e2b` or `gemma3n:e4b`.** That is a different family (Gemma 3n)
> and it is text-only on Ollama, so it cannot see images at all. The name collision with
> Gemma 4's E2B/E4B variants makes this an easy mistake.
>
> `gemma4:e4b-it-qat` is not a usable fallback on 4 GB of VRAM either. It crashed
> `llama-server` with a stack-based buffer overrun on a real 1280x720 capture,
> reproducibly. E2B is the working vision model here.

**2. Install the Python dependencies, with CPU-only torch**

Ultralytics pulls torch. Install it from the CPU index so all the VRAM stays free for
Gemma, which is the only part that needs it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn opencv-python requests
```

**3. Get the segmentation weights while you still have network**

`FastSAM-s.pt` must sit in the repo root. Ultralytics downloads it on first use, so run
the server once with wifi on, or fetch the file ahead of time. With wifi off and no local
copy, the import fails and the server never starts.

**4. Pick a camera (optional)**

Defaults to index 0. Set it if the tray camera is on a different index, or point it at a
stream URL.

```powershell
$env:ABSENT_CAMERA = "1"
```

**5. Run**

Every path in `app.py` is relative to the working directory, so launch from the repo
root.

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000**. The buttons are Count in, Count out, Reset. Both count
frames are written to `captures/` as evidence.

Endpoints: `GET /` (UI), `GET /feed` (MJPEG), `GET /state`, `POST /count/in`,
`POST /count/out`, `POST /reset`, `POST /ingest` (accepts a JPEG body from an external
camera source).

## Hardware

Built and measured on an HP Victus 15-fb0xxx: Ryzen 7 5800H, 16 GB RAM, RTX 3050 Ti with
4 GB VRAM, Windows 11. A consumer laptop, deliberately. The 4 GB ceiling is why the model
is E2B and why torch is CPU-only.

---

## Safety and scope

- **Decision support only.** Absent assists the count. It does not replace the nurse and
  it does not make a clinical determination. It flags an item as unaccounted for. A human
  decides what that means.
- **No diagnosis, no treatment recommendation.** The system never renders a judgment
  about a patient. It looks at a tray.
- **Synthetic and stand-in objects only.** No real patient data and no real OR footage
  were used at any point. Ordinary objects on a table stand in for instruments.
- **Nothing leaves the machine.** No API keys, no cloud inference, no telemetry.

## Known limitations

Stated plainly, because a count system that oversells itself is worse than no count
system.

- **A retained sponge inside the body is invisible to any camera.** It is retained
  precisely because it is hidden. Absent watches the instrument field, not the patient,
  and it does not claim otherwise.
- **Naming accuracy is unmeasured.** I measured determinism and latency. Whether the
  names are right on a real instrument tray, I have not measured.
- **Overlapping instruments are a weak point.** The whole frame goes to Gemma in one
  call, so two tools stacked on each other can come back as one name, or as none.
- **4 GB of VRAM forces E2B.** The larger E4B variant crashes on this machine, so there
  is no quality tier to escalate to.
- **Name drift between the two calls.** Count-in and count-out use different prompts, so
  temperature 0 does not guarantee the model echoes the count-in names character for
  character. The "trust the count-in list" rule keeps the failure direction safe, and the
  matching is exact string matching today.
