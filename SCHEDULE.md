# Build day schedule

Written 12:38 PM. Submissions lock **4:00 PM**. Presentations 4:00. Winners 5:30.
That is **202 minutes**, and about 45 of them are reserved for video.

## Read this first

**CORRECTION, 12:50 PM. The recording is mandatory. Earlier in this file it was called
insurance. That was wrong.**

The Kaggle competition page states: *"A valid submission must contain the following:
Kaggle Writeup, Attached Public Code Repository, Attached Live Demo (or Clonable
Notebook)."* The demo is one of three required components, and it accepts *"a URL or
files for your working demo (this can be a hosted web app, an interactive terminal
recording, or a fully functional Kaggle Notebook)."*

Absent is a local webcam plus a local Ollama server with wifi off. **You cannot satisfy
any of those three forms live.** There is no hostable URL, and no Kaggle Notebook can
reach your webcam. A screen recording, attached as a file or as an unlisted YouTube link,
is the only path to a *valid* submission.

**The 4:00 PM live presentation does not satisfy this.** It is a separate, parallel
deliverable. You need both. Organizer email, 07-26: *"your final demo has to run live"*
refers to the stage, not to Kaggle.

Consequence: the video block is no longer cuttable. It moved earlier and it is protected.

**The one thing that decides this project is still unmeasured.** Determinism and latency
are measured. Naming accuracy on a real tray is not. Nothing else on this list matters
until a real tray run happens, so it is first and it has a hard go/no-go.

**Guiding rule: submit something early and improve it.** A Kaggle writeup can be
re-submitted unlimited times. A placeholder submitted at 1:45 guarantees a non-zero
score. A perfect writeup submitted at 4:01 scores nothing.

---

## The schedule

**LOCK IS 3:45, NOT 4:00.** The posted event schedule says 3:45. `CONTEXT.md` recorded a
move to 4:00, but the board is the board. Assume 3:45 and lose nothing if it is wrong.
There is also a **3:00 mentor check-in** that will interrupt you. Plan for it.

| Time | Who | What | Done when |
|---|---|---|---|
| 1:20–1:45 | Enes | **The staged tray test.** Plain light surface, 6 to 8 clearly different objects, well separated, phone overhead, no shadow across the field. Count in, remove one, count out | The UI names the removed object under "unaccounted for" |
| **1:45** | Enes | **GO / NO-GO.** | See the fallback ladder below. Do not carry a broken pipeline past this point |
| 1:20–1:45 | collaborator | Name-drift fix in `check_against` (patch below). Then `pip freeze > requirements.txt` | A count-out that returns "scissor" for "scissors" does not report a phantom missing item |
| 1:45–2:00 | Enes | **Insurance submission.** Kaggle writeup, placeholder text, repo link, track = On-Device Private Health. **SAVE, then click SUBMIT** | Reload of the competition page shows a submitted entry |
| 1:45–2:00 | Enes | **Record the first working run the moment it works.** OBS Display Capture, not Game Bar | A raw recording of one successful count exists on disk |
| 2:00–2:20 | collaborator | Un-ignore `FastSAM-s.pt`, fix the README run command (says port 8000, app serves **HTTPS on 8443**), settle the missing `LICENSE` | A fresh clone runs with wifi off |
| 2:00–2:20 | Enes | Writeup body, **max 1,500 words**. Architecture and how Gemma 4 is used | Writeup final except the demo attachment |
| 2:20–2:35 | Enes | **Rehearsal 1, timed, out loud, standing.** `PITCH.md`, target 2:00 | You finished inside 2:00 with a stopwatch running |
| **2:35–3:20** | both | **Video block, 45 min. PROTECTED, this is a required submission component.** Record the full demo. Upload YouTube **unlisted**, or attach the MP4 as a file. Test in an incognito window | A stranger with the link can watch the working demo |
| 3:00 | Enes | Mentor check-in, if they come to you. Treat it as a free rehearsal | You said the pitch to a stranger once |
| 3:20–3:35 | Enes | Attach the demo to the writeup. **SAVE, then SUBMIT again** | Reload shows a submitted entry carrying writeup, repo, and demo |
| 3:35–3:45 | both | Buffer. Rehearsal 2 if nothing is broken | |
| **3:45** | | **SUBMISSIONS LOCK** | |
| 3:45–4:00 | both | Pre-stage checklist below | Every box ticked |
| **4:00** | Enes | Present live | |

**One writeup per team.** You merged with Hong Cheng Wang at 11:14 AM. Agree in the next
five minutes who owns it. Two writeups from one team is not something you want to
untangle at 3:40.

---

## Go / no-go at 1:15, fallback ladder

Take the first rung that works. Do not skip down the list out of optimism.

1. **Names are right.** Ship it as designed. Move on.
2. **Names are close but drift between count in and count out** (for example "scissors"
   then "shears"). Do not rewrite the pipeline. Add case-insensitive substring matching
   in `gemma.check_against` before falling back to exact string equality. 10 minutes.
3. **Names are wrong or invented.** Tighten `INVENTORY_PROMPT` first: name the object
   category you are actually showing, and say explicitly to return an empty array rather
   than guess. 10 minutes, one attempt only.
4. **Still wrong.** Change the scene, not the code. Higher contrast surface, brighter
   light, fewer objects, more separation. The model is fine, the image is hard.
5. **Nothing works by 2:00.** Demo the honest version: the live FastSAM overlay proves
   detection, and you present the count as measured determinism plus one prepared example.
   Say plainly what is working and what is not. Judges reward that far more than a
   demo that visibly lies.

**E4B is not on this ladder.** It crashes `llama-server` on this machine, reproducibly.
Do not spend a minute of build day trying to switch models.

---

## Verified demo killers, with the patches

Every one of these was reproduced on this machine, not guessed.

**1. Ollama evicts the model after 5 minutes idle. Cost: 2 minutes.**
`GET /api/ps` shows `expires_at` five minutes out. Sit through questions for six minutes
and the first count on stage takes **21 seconds**, not 4. In `gemma._ask`, next to
`"stream": False`:

```python
"keep_alive": -1,
```

**2. Name drift invents phantom missing instruments, at the exact climax. Cost: 10 min.**
Reproduced: count-in `['scissors','clamp']`, Gemma's count-out reply
`{"present": ["scissor","clamps"], "missing": []}`. Because `check_against` compares with
exact string equality, it returns **both** `present=['scissor','clamps']` and
`missing=['scissors','clamp']`. The screen shows two items unaccounted for when nothing
was removed. This fires precisely when the judge is watching. In `gemma.check_against`,
normalize before the set difference:

```python
n = lambda s: s.strip().lower().rstrip("s")
seen = {n(x) for x in present} | {n(x) for x in missing}
missing += [i for i in items if n(i) not in seen]
```

**3. `/cameras` permanently kills the live feed.** Confirmed: the probe loop opens index 0
while the capture thread already holds it, DirectShow hands the device over, and the
original holder never recovers. **Do not open that endpoint during the demo.** If you need
it, hardcode the index with `ABSENT_CAMERA` instead.

**4. The capture thread has no `try`/`except`.** One exception from `model.predict` and
the daemon thread dies silently. The feed freezes on the last good frame and counts keep
running against a stale image, with no error anywhere on screen. Wrap the loop body if
there is time. If not, know the symptom: **a frozen feed means restart the server.**

---

## If you are behind at 2:30, cut in this order

1. Rehearsal 2. Keep rehearsal 1.
2. The capture-thread `try`/`except`. Restart instead.
3. README polish. Judges read the Kaggle writeup first.
4. The video **re-record**. Ship the first take even if it is rough.

**Never cut:** the insurance submission, the video itself (it is a required submission
component), the `keep_alive` fix, the pre-stage warm-up, and the final SUBMIT click.

---

## Do not do these

- **Do not switch Gemma models.** E4B is broken here and E2B is already wired in.
- **Do not refactor the count pipeline.** It works, it is measured, and it is 4 seconds.
- **Do not chase the FastSAM mask thresholds** unless the overlay is visibly wrong on
  the actual demo scene. It is decoration for the live feel, and Gemma gets the whole
  uncropped frame regardless.
- **Do not add features.** Nothing new after 2:05.
- **Do not leave the Kaggle submission to the end.** People score zero every single
  hackathon because Save is not Submit.
- **Do not test with the built-in webcam pointed at your face.** That is what the 12:26
  capture was, and it measures nothing.
- **Do not open `/cameras` after the demo starts.** It steals the camera from the capture
  thread and the feed never comes back.
- **Do not record with Xbox Game Bar.** It captures only the focused window, stops on
  alt-tab, and will not record File Explorer or the desktop. Your demo is a browser window
  plus a physical tray. Use **OBS Display Capture**.
- **Do not upload the video as YouTube "private".** Judges get an access-denied page. It
  must be **unlisted**, and everything must open with no login and no paywall. Test it in
  an incognito window before you paste the link.

---

## Before you record, at 2:50

OBS on this machine was found misconfigured during the earlier session. Fix it before the
video block starts, not during it.

- Settings > Audio > **Desktop Audio = Default**, otherwise the recording is silent
- Settings > Video > both Base and Output resolution **1920x1080**
- Settings > Output > **Hybrid MP4**, which survives a crash mid-recording
- Use **Display Capture**, so the physical tray and the browser are both in frame

---

## Pre-stage checklist, 3:50

- [ ] **Warm the model.** One throwaway call so a cold-start crash cannot happen on
      stage. `llama-server` has died once already on a first cold call.
- [ ] Server running, UI loaded, one successful count already done since the last restart
- [ ] Objects staged on the tray, in frame, in focus, well lit
- [ ] **Wifi off**, and off in front of the room as part of the pitch
- [ ] Browser at the projector, display scaling checked from ten feet back. `Ctrl+=` is
      the fix on stage
- [ ] Video link open in a tab as the fallback if the live demo dies
- [ ] Kaggle entry confirmed submitted, not just saved
- [ ] Phone charged if it is the camera

---

## The 2 minute pitch, shape only

| t | beat |
|---|---|
| 0:00–0:20 | Retained surgical items are a never event and they still happen in 1 in 5,500 to 7,000 operations. The mitigation is a nurse counting out loud in the most distracting room in the hospital |
| 0:20–0:40 | The 2023 prior art needed a fine-tuned YOLOv8 and a hand-labeled instrument dataset collected during the hackathon |
| 0:40–1:00 | Gemma 4 names objects zero-shot, so the training requirement disappears. **Turn the wifi off here** |
| 1:00–1:30 | Count in. **Hand the tray to a judge and let them remove something** |
| 1:30–1:50 | Count out. It names exactly what is gone |
| 1:50–2:00 | This ran on a $700 laptop with no internet. The imagery never left the machine |

The judge removing the object is the strongest thing in the demo. No pre-recording
survives judge-supplied input, and everyone in the room knows it. Do not cut it.
