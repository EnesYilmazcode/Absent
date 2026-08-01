# Feature status

Last updated 1:55 PM. Deadline **4:00 PM EDT**, confirmed on the Kaggle page.

## The decision that has to happen first

**There are now two different products in this repo, and only one can be on stage.**

**Flow A, whole-scene count.** `POST /count/in` sends the whole frame to Gemma, which
names everything it sees. `POST /count/out` passes that list back and asks what is gone.
Two Gemma calls total, about 4 s each.

**Flow B, catalogued manifest.** `POST /catalog/add` is pressed once per instrument. You
hold the item up, Gemma names that one object, and a card with a cut-out thumbnail goes
into a manifest. `POST /verify` then checks the whole manifest against the live scene.
One Gemma call per item, about 4 s each, so eight items costs roughly 35 s of setup.

| | Flow A | Flow B |
|---|---|---|
| Setup time on stage | none | ~4 s per item |
| Failure mode | one bad inventory poisons the whole count | one bad name, the rest survive |
| Visual payload | a list of names | cards with real cut-out photos |
| Handles duplicates | yes, `Counter` based | yes |
| Matches `PITCH.md` | **yes** | no, the script would need rewriting |

**Recommendation: demo Flow B, pitch Flow A's insight.** The catalog cards are far more
convincing on a projector than a list of strings, and per-item naming is much more robust
than one all-or-nothing inventory. But **do the cataloguing before you walk up**, not on
stage, and keep the live moment as: judge removes an item, press Verify, the card flips to
unaccounted for.

If you disagree, pick Flow A and delete the catalog from the demo path. Either is fine.
**Showing both is not.** Two half-explained flows in two minutes reads as an unfinished
project.

---

## Built and working

| Feature | Route / file | Demo critical |
|---|---|---|
| Live class-agnostic segmentation overlay | `_capture_loop`, FastSAM | **yes** |
| Phone camera over USB or wifi, ~14 fps | `/phone`, `/ingest` | **yes** |
| Count zone, drawn by hand | `/zone`, `_roi` | yes |
| Whole-scene count in / count out | `/count/in`, `/count/out` | Flow A |
| Per-item catalog with cut-out cards | `/catalog/add`, `/catalog` | Flow B |
| Manifest verification | `/verify` | Flow B |
| Hold-it-up identify page | `/try`, `/identify` | no, but a great fallback |
| Duplicate-safe counting | `gemma.check_against` | **yes** |
| Fuzzy name matching | `gemma._match` | **yes** |
| Model pinned in memory | `keep_alive: -1` | **yes** |
| Self-signed cert for phone HTTPS | `make_cert.py` | yes |

## Not done

- [ ] **Kaggle writeup SUBMITTED.** A draft exists. A draft scores zero.
- [ ] Video recorded, uploaded, attached
- [ ] `requirements.txt`
- [ ] `FastSAM-s.pt` is gitignored, so a fresh clone cannot run offline
- [ ] `LICENSE` file, while the README badge claims AGPL-3.0
- [ ] README run command still says port 8000, the app serves HTTPS on 8443
- [ ] Rehearsal, out loud, timed

---

## Is this AI slop? No, and here is the evidence

Reviewed `gemma.py` and `app.py` at 1:55 PM. This is real code that does real work. The
comments explain *why*, and several of them record a bug that was actually hit:

- **`_UNREADABLE` sentinel** (`gemma.py:23`). An unreadable reply used to fall back to an
  empty dict, which reported **every** instrument unaccounted for. A hiccup would have
  thrown a red alarm on stage. Now it raises instead.
- **Counter-based counting** (`gemma.py:125`). Two clamps go in, one comes out, and a set
  difference reports all clear while an instrument is still inside the patient. This is a
  genuine correctness fix, and it is the kind of thing a judge will respect if you mention
  it.
- **`np.ptp(arr)` not `arr.ptp()`** (`app.py:370`). numpy 2.0 removed the method, and this
  ran on every catalog add, so the button 500'd every time. A real bug, found and fixed.
- **Fuzzy `_match`** (`gemma.py:91`). Plural, substring, then `difflib` at 0.75. Without
  it, "scissor" against "scissors" landed in neither list and got reported present and
  missing simultaneously.
- **Every handler that calls Gemma is wrapped** so a failure sets `state["error"]` instead
  of returning a 500 that leaves a dead button on stage.
- **Split connect and read timeouts**, `TIMEOUT = (3, 40)`. A dead Ollama fails in 3 s
  instead of hanging for two minutes.

That is a real engineering trail. Say so in the writeup: the "challenges you overcame"
section the rubric asks for is already written in these comments.

## What I would still fix, in priority order

1. **Pick one flow.** Above. Costs nothing but a decision.
2. **`inventory()` fails silently.** `gemma.py:87` falls back to `[]`, so an unreadable
   count-in shows an empty list rather than an error. The `named` vs `segmented`
   disagreement in the UI partly covers this, but if Flow A is the demo, make it raise
   like `check_against` does.
3. **`/catalog/add` has no duplicate guard.** Cataloguing the same object twice creates
   two cards, and `verify` will then expect two of them.
4. **`requirements.txt` and the weights.** Judges are told the repo is the source of
   truth. A repo nobody can run is a weak source of truth.
