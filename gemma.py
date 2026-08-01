"""Gemma 4 vision calls against a local Ollama server.

think=False and temperature=0 are both load-bearing, see CONTEXT.md:
thinking costs 120s per image, and nonzero temperature makes the object
names unstable enough to invent missing instruments.
"""

import base64
import difflib
import json
import re
from collections import Counter

import cv2
import requests

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b-it-qat"
SEED = 42
# Connect fast so a dead Ollama fails in seconds, but allow a slow first read.
TIMEOUT = (3, 40)

_UNREADABLE = object()   # sentinel: a bad reply must raise, not fall back

# Do not say "tray". Asked about a tray, E2B returns [] whenever it cannot see
# a literal surgical tray, which is every frame of our demo.
INVENTORY_PROMPT = (
    "List every portable object visible in this image as a JSON array "
    "of short lowercase names. Include tools, instruments, containers and small "
    "items lying on the surface. Do not list the surface itself, the table, the "
    "floor, walls, furniture, people, hands or clothing. "
    "Count duplicates: if two of the same object are present, write the name "
    "twice. Reply with the JSON array only."
)

CHECK_PROMPT = (
    "Earlier this scene held exactly these items:\n{items}\n\n"
    "Look at the image and decide which of those items are still visible.\n"
    'Reply with JSON only, in the form {{"present": [...], "missing": [...]}}, '
    "using the item names exactly as written above and listing every item in "
    "one of the two lists. If an item was listed twice above and only one is "
    "visible, write it once in present and once in missing."
)


def _encode(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode frame")
    return base64.b64encode(buf.tobytes()).decode()


def _ask(prompt, frame):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "images": [_encode(frame)],
        "stream": False,
        "think": False,
        # Ollama's default keep_alive is 5 minutes. Sit through six minutes of
        # questions and the first count on stage costs 21s instead of 4s.
        "keep_alive": -1,
        "options": {"temperature": 0, "seed": SEED},
    }
    r = requests.post(OLLAMA, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]


def _parse(text, fallback):
    """Gemma usually returns clean JSON but sometimes wraps it in a fence."""
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if not match:
        if fallback is _UNREADABLE:
            raise ValueError("no JSON in Gemma's reply, count not performed")
        return fallback
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        if fallback is _UNREADABLE:
            raise ValueError("malformed JSON from Gemma, count not performed")
        return fallback


def inventory(frame):
    """Name every object on the tray. Returns a list of names."""
    items = _parse(_ask(INVENTORY_PROMPT, frame), [])
    return [str(i).strip().lower() for i in items if str(i).strip()]


def _match(name, items):
    """Gemma does not echo names back exactly. Asked about "scissors" it will
    happily answer "scissor", and on exact equality that item lands in neither
    list and then gets reported present and missing at once. Map every answer
    back onto the count-in list instead."""
    name = re.sub(r"[^a-z0-9 ]", "", str(name).strip().lower())
    if not name:
        return None
    for item in items:
        a, b = name, item.lower()
        if a == b or a.rstrip("s") == b.rstrip("s") or a in b or b in a:
            return item
    close = difflib.get_close_matches(name, [i.lower() for i in items], 1, 0.75)
    return next((i for i in items if i.lower() == close[0]), None) if close else None


def check_against(frame, items):
    """Ask which of `items` are gone. Returns (present, missing)."""
    listed = "\n".join(f"- {i}" for i in items)
    # Never fall back silently here. An unreadable reply used to mean an empty
    # dict, which reported every instrument unaccounted for: a red alarm on
    # stage caused by a hiccup rather than by a missing instrument.
    result = _parse(_ask(CHECK_PROMPT.format(items=listed), frame), _UNREADABLE)

    # Gemma sometimes answers with a bare array instead of the two-key object.
    # Read that as the present list. Without this the count throws, or worse,
    # falls back to an empty dict and reports every instrument unaccounted for.
    if isinstance(result, list):
        result = {"present": result}
    elif not isinstance(result, dict):
        raise ValueError("unreadable reply from Gemma, count not performed")

    # Counted, not set-membership. Two clamps go in, one comes out, and a set
    # diff reports all clear while an instrument is still inside the patient.
    outstanding = Counter(items)
    present = []
    for answer in result.get("present", []):
        name = _match(answer, [i for i in outstanding if outstanding[i]])
        if name:
            outstanding[name] -= 1
            present.append(name)

    # Anything not clearly reported as still visible counts as missing. Failing
    # that way raises a false alarm; the other way loses an instrument.
    missing = list(outstanding.elements())
    return present, missing


HELD_PROMPT = (
    "Name the single object the person is holding up to the camera. "
    "Reply with just the name, one to three words, no punctuation, no sentence. "
    "If nobody is holding anything, reply exactly: none"
)


def identify_held(frame):
    """What is being held up right now. Used by the /try page."""
    return _ask(HELD_PROMPT, frame).strip().strip(".").lower()[:60]


def warm_up():
    """Cold starts have crashed llama-server once. Never do that on stage."""
    import numpy as np

    blank = np.zeros((64, 64, 3), dtype="uint8")
    _ask("Reply with the single word ready.", blank)
