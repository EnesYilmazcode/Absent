# Build day schedule

Written 12:38 PM. Submissions lock **4:00 PM**. Presentations 4:00. Winners 5:30.
That is **202 minutes**, and about 45 of them are reserved for video.

## Read this first

**Is a video required?** No. The rubric allows "live demo **or** video", and you are
presenting live anyway. The video is insurance against the demo failing on stage, not a
deliverable. It is still worth recording, because a frozen camera feed at 4:10 PM with no
backup costs you the Functionality score outright.

**The one thing that decides this project is still unmeasured.** Determinism and latency
are measured. Naming accuracy on a real tray is not. Nothing else on this list matters
until a real tray run happens, so it is first and it has a hard go/no-go.

**Guiding rule: submit something early and improve it.** A Kaggle writeup can be
re-submitted unlimited times. A placeholder submitted at 1:45 guarantees a non-zero
score. A perfect writeup submitted at 4:01 scores nothing.

---

## The schedule

| Time | Who | What | Done when |
|---|---|---|---|
| 12:40–1:15 | both | **The tray test.** Phone camera overhead on a light surface, 6 to 8 visually distinct objects, no overlap. Count in. Hand the tray away, remove one object. Count out. | The UI names the removed object under "unaccounted for", once, on real objects |
| **1:15** | Enes | **GO / NO-GO.** | See the fallback ladder below. Do not carry a broken pipeline past this point |
| 1:15–1:30 | both | Fix only what the tray test broke. Screen-record the first working run **the moment it works** (`Win + Alt + R`, lands in `Videos\Captures`) | A raw recording of one successful count exists on disk |
| 1:30–1:45 | Enes | **Insurance submission.** Kaggle writeup, placeholder text, repo link. **SAVE, then click SUBMIT.** They are two separate buttons | The competition page shows a submitted entry, confirmed by reload |
| 1:45–2:05 | collaborator | README fixes: run command says port 8000 but the app serves **HTTPS on 8443**, the endpoint list is missing `/phone`, `/cameras`, `/camera/{index}`, `/source`, and there is no `LICENSE` file despite the AGPL-3.0 badge | `README.md` run command matches what actually starts the server |
| 1:45–2:05 | Enes | Writeup body: problem, the zero-shot insight, measured numbers, safety, attribution | Writeup text is final except for the video link |
| 2:05–2:25 | both | Harden the demo. Warm-up call before every run. Practice the restart drill. Confirm the phone fallback path works | You can recover from a frozen feed in under 20 seconds without thinking |
| 2:25–2:40 | Enes | **Rehearsal 1, timed, out loud, standing.** Target 2:00 | You finished inside 2:00 with a stopwatch running |
| 2:40–2:55 | both | Buffer. Fix what rehearsal 1 exposed. If nothing broke, rehearse again | Nothing outstanding, or rehearsal 2 done |
| 2:55–3:40 | both | **Video block, 45 min.** Record the full demo. Re-record once if the first take is weak. Upload to YouTube **unlisted**. Test the link in an incognito window | A link a stranger can open plays the working demo |
| 3:40–3:52 | Enes | Paste the video link and repo link into the writeup. **SAVE, then SUBMIT again** | Reload shows the submitted entry carrying both links |
| 3:52–4:00 | both | Pre-stage checklist below | Every box ticked |
| **4:00** | Enes | Present | |

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

## If you are behind at 2:30, cut in this order

1. Rehearsal 2. Keep rehearsal 1.
2. The re-record. Ship the first video take even if it is rough.
3. README polish. The writeup matters more, judges read Kaggle first.
4. The video block entirely, down to a single unedited take of one successful count.

Never cut: the insurance submission, the pre-stage warm-up, and the final SUBMIT click.

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

---

## Pre-stage checklist, 3:52

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
