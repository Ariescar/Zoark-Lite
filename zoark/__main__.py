"""Zoark command line.

  python -m zoark simulate "Panthera tigris"
  python -m zoark order --structure examples/structure.json -n 12 -o out/demo
  python -m zoark models                                                                         (needs NEBIUS_API_KEY)
  python -m zoark extract --text examples/source.txt --save                                      (needs NEBIUS_API_KEY)
  python -m zoark order "front-heavy quadruped with a long tail for balance" -n 12 -o out/demo   (needs NEBIUS_API_KEY)
  python -m zoark device                                                                         (which CPU/GPU Newton runs on)
"""
import argparse
import json
import os
import re

from .body import make_body
from .order import load_species, run

SPECIES = os.environ.get("ZOARK_SPECIES", os.path.join(os.path.dirname(__file__), "..", "species"))


def _save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(prog="zoark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate"); s.add_argument("species"); s.add_argument("--seconds", type=float, default=2.0)
    o = sub.add_parser("order"); o.add_argument("text", nargs="?"); o.add_argument("--structure")
    o.add_argument("-n", type=int, default=12); o.add_argument("-o", "--out", default="out/demo"); o.add_argument("--seconds", type=float, default=2.0)
    o.add_argument("--save-structure", help="also write the Nemotron structure to this JSON file")
    e = sub.add_parser("extract"); e.add_argument("--text"); e.add_argument("--image")
    e.add_argument("--save", action="store_true", help="add the result to species/ when it has the needed numbers")
    e.add_argument("--out", help="also write the raw Nemotron result to this JSON file")
    sub.add_parser("models")
    sub.add_parser("device")
    a = ap.parse_args()

    if a.cmd == "models":
        from .reader import BASE_URL, nemotron_models, pick_models
        ids = nemotron_models()
        text, vision = pick_models(ids)
        print(json.dumps({"endpoint": BASE_URL, "nemotron_models": ids, "text_model": text, "vision_model": vision}, indent=1))
        return
    if a.cmd == "device":
        import warp as wp
        import newton
        wp.init()
        print(json.dumps({"newton": getattr(newton, "__version__", ""), "warp": wp.config.version,
                          "device": str(wp.get_device()), "cuda_devices": [str(d) for d in wp.get_cuda_devices()]}, indent=1))
        return

    species = load_species(SPECIES)
    if a.cmd == "simulate":
        from .physics import stand_test
        print(json.dumps(stand_test(make_body(species[a.species]), seconds=a.seconds), indent=1))
    elif a.cmd == "extract":
        from .reader import extract
        text = open(a.text, encoding="utf-8").read() if a.text else None
        d = extract(text, a.image)
        print(json.dumps(d, indent=1, ensure_ascii=False))
        if a.out:
            _save(d, a.out)
        if a.save:
            if not d.get("usable"):
                raise SystemExit(f"not saved: missing {d.get('missing')}")
            rec = {k: d[k] for k in ("name", "common_name", "shoulder_height_m", "head_body_length_m", "tail_length_m", "quote", "_nemotron") if k in d}
            _save(rec, os.path.join(SPECIES, re.sub(r"[^a-z0-9]+", "_", d["name"].lower()).strip("_") + ".json"))
    else:
        if a.structure:
            structure = json.load(open(a.structure, encoding="utf-8"))
        else:
            if not a.text:
                raise SystemExit("Give a request text or --structure.")
            from .reader import compose
            structure = compose(a.text, list(species))
            if a.save_structure:
                _save(structure, a.save_structure)
        print("structure:", json.dumps(structure, ensure_ascii=False))
        m = run(structure, species, a.n, a.out, seconds=a.seconds)
        print(f"\n{m['passed']}/{m['simulated']} variants passed -> {a.out}")


if __name__ == "__main__":
    main()
