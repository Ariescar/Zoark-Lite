# Zoark-Lite

**Biomimetic synthetic data for robots: from a robot team's request to physics-validated body variants.**

Zoark-Lite is the open, lightweight engine behind [ZOARK](https://zoark.org). A robot team describes a body it wants to test; NVIDIA Nemotron turns the request into a creature structure; Zoark-Lite generates many body variants and tests every one in **NVIDIA Newton**; the variants that stand are packed into a dataset (MJCF models + physics results) for reinforcement and imitation learning.

Built for the **Nebius × NVIDIA Global AI Hackathon** (Best Apps and Agents). Live site: https://zoark.org · Demo video: *(YouTube link)*

## Pipeline

| Step | What happens | NVIDIA / Nebius |
|---|---|---|
| Extract | Read a text or an animal photo and pull out body measurements | NVIDIA Nemotron on Nebius Token Factory |
| Compose | Turn a plain-language request into a creature structure | NVIDIA Nemotron on Nebius Token Factory |
| Build | Lay out a parametric quadruped body from the measurements | — |
| Prove | Stand test of every variant: does it stay up and level, where is its centre of mass | NVIDIA Newton (GPU on Nebius AI Cloud, or CPU) |
| Package | Variants that pass → MJCF + body + simulation JSON + results.csv | — |

## Quick start

```bash
git clone https://github.com/Ariescar/Zoark-Lite
cd Zoark-Lite
pip install -r requirements.txt

# Newton stand test of one species (no key needed)
python -m zlite simulate "Panthera tigris"

# a full order from a ready-made structure (no key needed)
python -m zlite order --structure examples/structure.json -n 12 -o out/demo
```

With a Nebius Token Factory key, Nemotron reads and composes:

```bash
export NEBIUS_API_KEY=...                                    # https://tokenfactory.nebius.com
export NEMOTRON_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B  # default
export NEMOTRON_VISION_MODEL=<Nemotron model on Token Factory that reads images>   # for --image

python -m zlite extract --text examples/source.txt
python -m zlite extract --image photo.jpg
python -m zlite order "front-heavy quadruped with a long tail for balance" -n 20 -o out/order1
```

Add species by dropping a JSON file into `species/` (`name`, `shoulder_height_m`, `head_body_length_m`, `tail_length_m`), or create one with `extract`.

### On Nebius AI Cloud

```bash
docker build -t zoark-lite .
docker run --gpus all -e NEBIUS_API_KEY=$NEBIUS_API_KEY -v $PWD/out:/app/out zoark-lite
```
NVIDIA Warp picks the GPU automatically; without a GPU it runs on CPU.

### Output

```
out/demo/
  manifest.json          structure + counts
  results.csv            every variant: scales, mass, stands, tilt, height drop, tip margin, support area
  variants/<name>/
    model.xml            MJCF (loads in MuJoCo and Newton)
    body.json            the body variant
    simulation.json      Newton run: trunk height trace, tilt, centre of mass, support polygon
```

## How NVIDIA and Nebius are used

- **NVIDIA Nemotron via Nebius Token Factory** (`zlite/reader.py`): `extract()` reads source text or a photo into measurements; `compose()` turns a request into a creature structure. Nemotron is asked to interpret only; every number must come from the source.
- **Where Token Factory helped:** one OpenAI-compatible endpoint, no GPU to manage, so the reading step worked from day one in both this engine and the zoark.org site, billed per request.
- **NVIDIA Newton** (`zlite/physics.py`): each variant becomes an articulated body (free trunk, PD-held hinge joints) dropped on a ground plane and solved with Newton's MuJoCo solver; the run decides pass/fail.
- **Nebius AI Cloud**: GPU instances run the batch simulations (Dockerfile above).

## Scope

Zoark-Lite uses a simple capsule template driven by a few external measurements. zoark.org offers higher-fidelity digital twins as a service.

## Licence

MIT (see `LICENSE`). Third-party components: `THIRD_PARTY_NOTICES.md`. Species values in `species/` are approximate public field-guide figures.

Alsomind Tech Co., Ltd. · Taipei, Taiwan
