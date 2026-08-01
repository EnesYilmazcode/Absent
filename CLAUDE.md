# Working rules for Absent

Read `CONTEXT.md` first, every session. It is the source of truth for what we're
building, what's decided, and what's rejected.

## Commit and push constantly — this is the top rule

Two people are building this repo, each with their own Claude Code agent. The repo
is the only shared memory between those agents. An uncommitted change is a decision
the other side cannot see.

- Commit after **every** meaningful change. Do not batch work into one big commit.
- **Push immediately after every commit.** A local commit helps nobody.
- `git pull --rebase` before you start and before you push, so histories stay linear.
- Write real commit messages that say what changed and why — the other agent reads
  the log to reconstruct context.
- When a decision changes, update `CONTEXT.md` in the same commit as the code that
  reflects it.
- Do not ask for permission to commit and push routine work. Just do it.

## Everything else

- No secrets in the repo. The whole system is local; there should be nothing to hide.
- No cloud inference. On-device only — cloud calls disqualify the track.
- Anything an agent should not touch goes in `.gitignore` and gets noted in
  `CONTEXT.md`.
