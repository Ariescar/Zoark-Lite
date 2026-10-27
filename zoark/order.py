"""Order → creature structure → variants → Newton stand tests → dataset folder."""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import time

from .body import make_body
from .mjcf import to_mjcf

KEYS = ("legs", "trunk", "tail", "front")


def load_species(folder: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(f))
        out[d["name"]] = d
    return out


def run(structure: dict, species: dict, n: int, out_dir: str, seconds: float = 2.0, seed: int = 0) -> dict:
    from .physics import stand_test
    base = species[structure["base"]]
    scale = {k: float(v) for k, v in structure.get("scale", {}).items() if k in KEYS + ("size",)}
    ranges = structure.get("ranges", {k: [0.85, 1.25] for k in KEYS})
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows, passed, t0 = [], 0, time.time()
    for i in range(n):
        k = {key: round(scale.get(key, 1.0) * rng.uniform(*ranges.get(key, [1, 1])), 3) for key in KEYS}
        body = make_body(base, size=scale.get("size", 1.0), **k)
        body.name = f"{structure.get('name', 'creature').replace(' ', '-')}-{i:03d}"
        sim = stand_test(body, seconds=seconds)
        rows.append({"variant": body.name, **{f"scale_{x}": k[x] for x in KEYS}, **{x: sim[x] for x in
                     ("stands", "mass_kg", "trunk_tilt_deg", "height_drop_ratio", "tip_margin_m", "support_area_m2")}})
        print(f"{body.name}: {'PASS' if sim['stands'] else 'fail'}  tilt {sim['trunk_tilt_deg']}°", flush=True)
        if sim["stands"]:
            passed += 1
            d = os.path.join(out_dir, "variants", body.name)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "model.xml"), "w").write(to_mjcf(body))
            json.dump(body.to_dict(), open(os.path.join(d, "body.json"), "w"), indent=1)
            json.dump(sim, open(os.path.join(d, "simulation.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    manifest = {"structure": structure, "simulated": n, "passed": passed, "physics": "NVIDIA Newton stand test",
                "engine": sim["engine"], "device": sim["device"], "seconds_wall": round(time.time() - t0, 1)}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return manifest
