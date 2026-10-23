#!/usr/bin/env bash
# Runs inside a Nebius AI Cloud Serverless job (GPU). Builds and stand-tests every variant in NVIDIA Newton.
# The structure comes from the Nemotron run committed in examples/ (no API key needed in the cloud).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q --root-user-action=ignore -r requirements.txt
echo "=== device ==="
python -m zlite device
STRUCT="${STRUCT:-examples/nemotron_compose.json}"
[ -f "$STRUCT" ] || STRUCT=examples/structure.json
python -m zlite order --structure "$STRUCT" -n "${N:-64}" -o out/cloud
echo "=== manifest.json ==="; cat out/cloud/manifest.json
echo; echo "=== results.csv ==="; cat out/cloud/results.csv
