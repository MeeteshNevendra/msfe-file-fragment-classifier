#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python -m msfe_fragment_classifier predict \
  --model-dir "$ROOT/outputs/demo_model" \
  --input-npz "$ROOT/data/demo_fragments/test.npz" \
  --out "$ROOT/outputs/demo_predictions.csv"
