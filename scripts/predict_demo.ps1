$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"

python -m msfe_fragment_classifier predict `
  --model-dir (Join-Path $Root "outputs\demo_model") `
  --input-npz (Join-Path $Root "data\demo_fragments\test.npz") `
  --out (Join-Path $Root "outputs\demo_predictions.csv")
