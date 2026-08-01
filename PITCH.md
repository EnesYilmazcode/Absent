# The pitch

## What to call it in one breath

The name stays **Absent**. It is already in the UI, the README and the repo, and renaming
at 1:20 PM buys nothing.

When you introduce yourself, use one of these. Shortest first.

**Three words:**
> "Absent. Surgical instrument counting."

**One line, the one to default to:**
> "I'm Enes, and I'm building Absent. It makes sure nothing gets left inside a patient."

**One line, if the room is technical:**
> "I'm Enes, I'm building Absent, a surgical instrument count that runs entirely
> on-device."

**One line, if you want the problem to land first:**
> "I'm Enes. Surgeons leave instruments inside patients a few thousand times a year. I
> built the count that catches it."

Do not say "AI-powered", "leveraging Gemma", or "computer vision solution". Say what it
does. The judges have heard forty pitches by the time you walk up.

**The sentence that carries the whole project.** If you only get one line out, this is it:

> "The 2023 version of this needed a hand-labeled dataset of surgical instruments. Gemma
> names them zero-shot, so it needs none, and it works on instruments nobody trained it
> on."

---

## The 2 minute script

Say it out loud twice before you present. Reading it silently does not count. The bracket
lines are stage directions, not words.

> **[0:00, show the tray on screen]**
>
> A retained surgical item is when something gets sewn inside a patient. A sponge, a
> clamp, a needle. It's classified as a "never event", which means it is supposed to be
> impossible, and it still happens in roughly one in every five to seven thousand
> operations.
>
> The way we prevent it today is a nurse counting instruments out loud, twice, from
> memory, in the most distracting room in the hospital.
>
> **[0:25]**
>
> This is Absent. A camera watches the instrument field. At count-in it names everything
> it sees. At count-out it checks that same list and flags anything it can't account for.
>
> **[0:40, turn the wifi off, visibly]**
>
> I'm turning the wifi off now. It'll keep working.
>
> Operating room footage is about the most sensitive data there is. It cannot go to a
> cloud API, so none of this does. The model is running on this laptop. A seven hundred
> dollar laptop, not a server.
>
> **[0:55, hit Count in]**
>
> Count in.
>
> **[1:05, hand the tray to a judge]**
>
> Take this. Remove something, don't tell me what.
>
> **[1:20, take it back, hit Count out]**
>
> Count out.
>
> **[1:30, the missing item appears in red]**
>
> There it is.
>
> **[1:40, the part that wins it]**
>
> The reason this is a Gemma project and not a detection project: the system this builds
> on won HackOHI/O in 2023, and it needed a fine-tuned YOLOv8. They spent their hackathon
> collecting and hand-labeling a dataset of surgical instruments, because a detector only
> knows the classes you trained it on.
>
> Gemma 4 is a vision-language model. It names objects zero-shot. No dataset, no labeling,
> no training run, and it generalizes to instruments nobody has ever trained a detector
> on. Take Gemma out and this project goes back to needing a labeled dataset it doesn't
> have.
>
> **[2:00]**
>
> Nothing left behind. Thank you.

**If you are running long, cut in this order:** the "seven hundred dollar laptop" aside,
then the sponge/clamp/needle list, then the second half of the never-event explanation.
**Never cut** the wifi moment or the judge removing an object.

---

## The two beats that actually win

1. **Turning the wifi off in front of the room.** Everybody claims privacy. You are the
   only one who can prove it standing there.
2. **Handing the tray to a judge.** No pre-recorded demo survives judge-supplied input,
   and every person in that room knows it. This single move is worth more than any slide.

Do them both. They are cheap and nobody else will.

---

## Questions you will get, and the answers

**"Why not just fine-tune a detector? It'd be more accurate."**
> It would, on the instruments you trained it on. Every hospital has a different tray, and
> a detector that hasn't seen an instrument can't count it. That's the failure mode I care
> about, and zero-shot naming is the only thing that addresses it.

**"How accurate is it?"**
> I measured determinism and latency, not accuracy. Three runs on the same frame come back
> byte-identical, and a count takes about four seconds. I have not measured naming accuracy
> against a labeled tray, and I'm not going to claim a number I didn't measure.

**"What if it misses something?"**
> It fails toward flagging. If the model doesn't confirm an item is present, that item goes
> in the unaccounted-for list. A false alarm costs a nurse ten seconds. A miss is the thing
> we're trying to prevent.

**"Can it see a sponge inside the patient?"**
> No, and nothing can. A retained sponge is retained precisely because it's hidden. This
> watches the instrument field, not the patient. It's the count that's broken, not the
> imaging.

**"Is this a medical device? Are you diagnosing?"**
> No. It's decision support for a count. It never says anything about the patient. It says
> "this object was on the tray and now I can't see it", and a human decides what that
> means.

**"What's actually yours versus the 2023 project?"**
> Their system is a fine-tuned YOLOv8 for instrument tracking, AGPL-licensed, and I credit
> it in the README. What's mine is replacing the fine-tune with zero-shot naming so the
> training requirement disappears entirely, plus the count-in and count-out logic on top.

**"Why is it slow?"**
> It isn't, for what it does. Four seconds, twice per operation. Real counts happen at
> defined moments, before incision and before closure, not continuously. The architecture
> matches the workflow instead of fighting it.

---

## Before you walk up

- Warm the model with one throwaway count. Cold start is 21 seconds, warm is 4.
- One successful count already done since the last server restart.
- Objects staged, in frame, in focus, lit, no shadow across the field.
- Wifi still on, so you can turn it off as a beat.
- Video link open in a tab, in case the live demo dies.
- Say the first sentence out loud once before you go up. It is always the hardest one.
