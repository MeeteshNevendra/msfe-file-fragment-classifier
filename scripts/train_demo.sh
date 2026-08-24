#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python "$ROOT/examples/create_demo_dataset.py" \
  --out-dir "$ROOT/data/demo_fragments" \
  --seq-len 512 \
  --classes 3 \
  --samples-per-class 90

python -m msfe_fragment_classifier train \
  --data-dir "$ROOT/data/demo_fragments" \
  --out-dir "$ROOT/outputs/demo_model" \
  --epochs 1 \
  --batch-size 32
