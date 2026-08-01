"""Gemma 4 vision calls against a local Ollama server.

think=False and temperature=0 are both load-bearing, see CONTEXT.md:
thinking costs 120s per image, and nonzero temperature makes the object
names unstable enough to invent missing instruments.
"""

import base64
import json
import re

import cv2
import requests

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b-it-qat"
SEED = 42
TIMEOUT = 120

# Do not say "tray". Asked about a tray, E2B returns [] whenever it cannot see
# a literal surgical tray, which is every frame of our demo.
INVENTORY_PROMPT = (
    "List every distinct portable object visible in this image as a JSON array "
    "of short lowercase names. Include tools, instruments, containers and small "
    "items lying on the surface. Do not list the surface itself, the table, the "
    "floor, walls, furniture, people, hands or clothing. "
    "Use one name per object. Reply with the JSON array only."
)

CHECK_PROMPT = (
    "Earlier this scene held exactly these items:\n{items}\n\n"
    "Look at the image and decide which of those items are still visible.\n"
    'Reply with JSON only, in the form {{"present": [...], "missing": [...]}}, '
    "using the item names exactly as written above and listing every item in "
    "one of the two lists."
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
        return fallback
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback


def inventory(frame):
    """Name every object on the tray. Returns a list of names."""
    items = _parse(_ask(INVENTORY_PROMPT, frame), [])
    return [str(i).strip().lower() for i in items if str(i).strip()]


def check_against(frame, items):
    """Ask which of `items` are gone. Returns (present, missing)."""
    listed = "\n".join(f"- {i}" for i in items)
    result = _parse(_ask(CHECK_PROMPT.format(items=listed), frame), {})
    present = [str(i).strip().lower() for i in result.get("present", [])]
    missing = [str(i).strip().lower() for i in result.get("missing", [])]

    # Trust the count-in list over the model: anything it failed to mention
    # is unaccounted for, which is the safe direction to fail in.
    seen = set(present) | set(missing)
    missing += [i for i in items if i not in seen]
    return present, missing


def warm_up():
    """Cold starts have crashed llama-server once. Never do that on stage."""
    import numpy as np

    blank = np.zeros((64, 64, 3), dtype="uint8")
    _ask("Reply with the single word ready.", blank)
