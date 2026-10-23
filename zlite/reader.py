"""NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API).

Two jobs:
  extract()  read a text and/or an animal photo and return the measurements a body needs, as JSON.
  compose()  read a robot team's order and return a structure for a new creature:
             which species to start from and how to scale the body.

Set NEBIUS_API_KEY. Model ids are read from Token Factory's own model list (GET /models), so the
code follows whatever Nemotron models your key can use. Pin one with NEMOTRON_MODEL /
NEMOTRON_VISION_MODEL if you want.

Every result carries a "_nemotron" record: model id, endpoint, token usage and time of the call.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import mimetypes
import os
import re

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
TEXT_MODEL = os.environ.get("NEMOTRON_MODEL", "")
VISION_MODEL = os.environ.get("NEMOTRON_VISION_MODEL", "")
MAX_TOKENS = int(os.environ.get("NEMOTRON_MAX_TOKENS", "4096"))   # room for reasoning + answer

_SKIP = ("embed", "reward", "guard", "safety", "rerank", "parse", "retriever")
_TEXT_PREF = ("nano", "lightning", "super", "ultra")               # fast first, big last


def client():
    from openai import OpenAI
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise SystemExit("Set NEBIUS_API_KEY (Nebius Token Factory) first.")
    return OpenAI(base_url=BASE_URL, api_key=key)


# ---------------------------------------------------------------- model choice

def _is_vision(mid: str) -> bool:
    return re.search(r"(^|[-_/.])(vl|omni|vision)([-_/.]|$)", mid.lower()) is not None


def nemotron_models(cl=None) -> list[str]:
    """Nemotron chat models this key can use, from Token Factory's GET /models."""
    cl = cl or client()
    ids = [m.id for m in cl.models.list().data]
    return sorted(i for i in ids if "nemotron" in i.lower() and not any(s in i.lower() for s in _SKIP))


def pick_models(ids: list[str]) -> tuple[str, str]:
    """(text model, vision model) from a list of Nemotron ids. Either may be '' if none fits."""
    text = [i for i in ids if not _is_vision(i)]
    vision = [i for i in ids if _is_vision(i)]
    rank = lambda i: next((n for n, k in enumerate(_TEXT_PREF) if k in i.lower()), len(_TEXT_PREF))
    text.sort(key=rank)
    vision.sort(key=rank)
    return (text[0] if text else (vision[0] if vision else "")), (vision[0] if vision else "")


def resolve(need_vision: bool = False, cl=None) -> str:
    if need_vision and VISION_MODEL:
        return VISION_MODEL
    if not need_vision and TEXT_MODEL:
        return TEXT_MODEL
    text, vision = pick_models(nemotron_models(cl))
    model = vision if need_vision else text
    if not model:
        kind = "image-reading " if need_vision else ""
        raise SystemExit(f"No {kind}Nemotron model found on Token Factory for this key. "
                         f"Run `python -m zlite models` and set NEMOTRON_{'VISION_' if need_vision else ''}MODEL.")
    return model


# ---------------------------------------------------------------- calling

def _json(text: str) -> dict:
    """Last JSON object in a reply. Reasoning blocks (<think>...</think>) are dropped first."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    dec, found, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                found = obj
            i = end
        except ValueError:
            i += 1
    if found is None:
        raise ValueError("Nemotron did not return JSON: " + text[:300])
    return found


def _chat(model: str, system: str, content, temperature: float, cl=None) -> dict:
    cl = cl or client()
    r = cl.chat.completions.create(model=model, temperature=temperature, max_tokens=MAX_TOKENS,
                                   messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": content}])
    msg = r.choices[0].message
    out = _json(msg.content or "")
    u = getattr(r, "usage", None)
    out["_nemotron"] = {"model": r.model or model, "endpoint": BASE_URL,
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    return out


# ---------------------------------------------------------------- extract

EXTRACT_PROMPT = """Extract external body measurements of ONE animal species. Return JSON only:
{"name": "<scientific name>", "common_name": "...", "shoulder_height_m": <n>, "head_body_length_m": <n>, "tail_length_m": <n>,
 "quote": "<the words or image cue each number came from>"}
Use only numbers given in the source; omit what is missing; convert to metres."""

LIMITS = {"shoulder_height_m": (0.02, 6.0), "head_body_length_m": (0.03, 12.0), "tail_length_m": (0.0, 12.0)}


def check_species(d: dict) -> dict:
    """Keep only plausible numbers; a species needs a name, shoulder height and head-body length."""
    for k, (lo, hi) in LIMITS.items():
        if k in d:
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                d.pop(k); continue
            if lo <= v <= hi:
                d[k] = v
            else:
                d.pop(k)
    missing = [k for k in ("name", "shoulder_height_m", "head_body_length_m") if not d.get(k)]
    d["usable"] = not missing
    if missing:
        d["missing"] = missing
    return d


def extract(text: str | None = None, image_path: str | None = None, cl=None) -> dict:
    """Pull measurements out of a source text and/or an animal photo."""
    content = []
    if text:
        content.append({"type": "text", "text": "SOURCE:\n" + text})
    if image_path:
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    if not content:
        raise SystemExit("Give --text and/or --image.")
    model = resolve(need_vision=bool(image_path), cl=cl)
    return check_species(_chat(model, EXTRACT_PROMPT, content, 0.0, cl))


# ---------------------------------------------------------------- compose

COMPOSE_PROMPT = """Turn a robot team's request into a creature structure. Start from one species in AVAILABLE.
Return JSON only:
{"name": "...", "base": "<species from AVAILABLE>", "scale": {"legs": n, "trunk": n, "tail": n, "front": n, "size": n},
 "ranges": {"legs": [lo, hi], "trunk": [lo, hi], "tail": [lo, hi], "front": [lo, hi]}, "why": "..."}
Scales are multipliers around 1.0 (allowed 0.5 to 2.0)."""

KEYS = ("legs", "trunk", "tail", "front")


def check_structure(s: dict, available: list[str]) -> dict:
    """Make a composed structure safe to run: known base species, scales and ranges inside 0.5-2.0."""
    base = str(s.get("base", ""))
    match = next((a for a in available if a.lower() == base.lower()), None) or \
        next((a for a in available if base and (base.lower() in a.lower() or a.lower() in base.lower())), None)
    if not match:
        raise ValueError(f"Nemotron picked base '{base}', not one of {available}")
    s["base"] = match
    clamp = lambda v: round(min(2.0, max(0.5, float(v))), 3)
    s["scale"] = {k: clamp(v) for k, v in (s.get("scale") or {}).items() if k in KEYS + ("size",)}
    rng = {}
    for k, v in (s.get("ranges") or {}).items():
        if k in KEYS and isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = sorted((clamp(v[0]), clamp(v[1])))
            rng[k] = [lo, hi]
    s["ranges"] = {k: rng.get(k, [0.85, 1.25]) for k in KEYS}
    s.setdefault("name", "creature")
    return s


def compose(order: str, available: list[str], cl=None) -> dict:
    model = resolve(cl=cl)
    s = _chat(model, COMPOSE_PROMPT, f"AVAILABLE: {available}\nREQUEST: {order}", 0.2, cl)
    s["request"] = order
    return check_structure(s, available)
