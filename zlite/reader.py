"""NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API).

Two jobs:
  extract()  read a text and/or an animal photo and return the measurements a body needs, as JSON.
  compose()  read a robot team's order and return a structure for a new creature:
             which species to start from and how to scale the body.

Set NEBIUS_API_KEY. Model ids are configurable; defaults are Nemotron models on Token Factory.
"""
from __future__ import annotations

import base64
import json
import os
import re

from openai import OpenAI

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
TEXT_MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
VISION_MODEL = os.environ.get("NEMOTRON_VISION_MODEL", "")   # a Nemotron model that reads images, e.g. a Nano Omni id listed by GET /models


def client() -> OpenAI:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise SystemExit("Set NEBIUS_API_KEY (Nebius Token Factory) first.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model did not return JSON: " + text[:200])
    return json.loads(m.group(0))


EXTRACT_PROMPT = """Extract external body measurements of ONE animal species. Return JSON only:
{"name": "<scientific name>", "common_name": "...", "shoulder_height_m": <n>, "head_body_length_m": <n>, "tail_length_m": <n>,
 "quote": "<the words or image cue each number came from>"}
Use only numbers given in the source; omit what is missing; convert to metres."""


def extract(text: str | None = None, image_path: str | None = None) -> dict:
    """Pull measurements out of a source text and/or an animal photo."""
    content = []
    if text:
        content.append({"type": "text", "text": "SOURCE:\n" + text})
    model = TEXT_MODEL
    if image_path:
        if not VISION_MODEL:
            raise SystemExit("Set NEMOTRON_VISION_MODEL to a Nemotron model on Token Factory that reads images.")
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        model = VISION_MODEL
    r = client().chat.completions.create(model=model, temperature=0.0, max_tokens=1200,
                                         messages=[{"role": "system", "content": EXTRACT_PROMPT},
                                                   {"role": "user", "content": content}])
    return _json(r.choices[0].message.content)


COMPOSE_PROMPT = """Turn a robot team's request into a creature structure. Start from one species in AVAILABLE.
Return JSON only:
{"name": "...", "base": "<species from AVAILABLE>", "scale": {"legs": n, "trunk": n, "tail": n, "front": n, "size": n},
 "ranges": {"legs": [lo, hi], "trunk": [lo, hi], "tail": [lo, hi], "front": [lo, hi]}, "why": "..."}
Scales are multipliers around 1.0."""


def compose(order: str, available: list[str]) -> dict:
    r = client().chat.completions.create(model=TEXT_MODEL, temperature=0.2, max_tokens=800,
                                         messages=[{"role": "system", "content": COMPOSE_PROMPT},
                                                   {"role": "user", "content": f"AVAILABLE: {available}\nREQUEST: {order}"}])
    return _json(r.choices[0].message.content)
