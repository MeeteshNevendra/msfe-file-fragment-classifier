$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"

python (Join-Path $Root "examples\create_demo_dataset.py") `
  --out-dir (Join-Path $Root "data\demo_fragments") `
  --seq-len 512 `
  --classes 3 `
  --samples-per-class 90

python -m msfe_fragment_classifier train `
  --data-dir (Join-Path $Root "data\demo_fragments") `
  --out-dir (Join-Path $Root "outputs\demo_model") `
  --epochs 1 `
  --batch-size 32
