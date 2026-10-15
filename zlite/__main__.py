"""Zoark-Lite command line.

  python -m zlite simulate "Panthera tigris"
  python -m zlite order --structure examples/structure.json -n 12 -o out/demo
  python -m zlite order "front-heavy quadruped with a long tail for balance" -n 12 -o out/demo   (needs NEBIUS_API_KEY)
  python -m zlite extract --text examples/source.txt                                             (needs NEBIUS_API_KEY)
"""
import argparse
import json
import os

from .body import make_body
from .order import load_species, run

SPECIES = os.environ.get("ZLITE_SPECIES", os.path.join(os.path.dirname(__file__), "..", "species"))


def main():
    ap = argparse.ArgumentParser(prog="zlite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate"); s.add_argument("species"); s.add_argument("--seconds", type=float, default=2.0)
    o = sub.add_parser("order"); o.add_argument("text", nargs="?"); o.add_argument("--structure")
    o.add_argument("-n", type=int, default=12); o.add_argument("-o", "--out", default="out/demo"); o.add_argument("--seconds", type=float, default=2.0)
    e = sub.add_parser("extract"); e.add_argument("--text"); e.add_argument("--image")
    a = ap.parse_args()
    species = load_species(SPECIES)
    if a.cmd == "simulate":
        from .physics import stand_test
        print(json.dumps(stand_test(make_body(species[a.species]), seconds=a.seconds), indent=1))
    elif a.cmd == "extract":
        from .reader import extract
        print(json.dumps(extract(open(a.text).read() if a.text else None, a.image), indent=1))
    else:
        if a.structure:
            structure = json.load(open(a.structure))
        else:
            from .reader import compose
            structure = compose(a.text, list(species))
        print("structure:", json.dumps(structure))
        m = run(structure, species, a.n, a.out, seconds=a.seconds)
        print(f"\n{m['passed']}/{m['simulated']} variants passed -> {a.out}")


if __name__ == "__main__":
    main()
