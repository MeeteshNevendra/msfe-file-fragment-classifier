# Multi-Scale File Fragment Classifier

This repository contains a byte-level file fragment classifier built around a multi-scale neural ensemble and a histogram gradient boosting meta-classifier. The workflow supports training on fixed-length byte fragments, evaluating held-out test data, and predicting labels for new fragments or raw binary files.

## Repository Layout

```text
msfe-file-fragment-classifier/
├── README.md
├── data/
│   └── README.md
├── docs/
│   ├── approach.md
│   ├── dataset_format.md
│   └── reproducibility.md
├── examples/
│   └── create_demo_dataset.py
├── scripts/
│   ├── predict_demo.ps1
│   ├── predict_demo.sh
│   ├── train_demo.ps1
│   └── train_demo.sh
└── src/
    └── msfe_fragment_classifier/
        ├── __init__.py
        ├── __main__.py
        └── cli.py
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick Demo

Create a small byte-fragment dataset:

```powershell
python examples\create_demo_dataset.py --out-dir data\demo_fragments --seq-len 512 --classes 3 --samples-per-class 90
```

Train the model:

```powershell
python -m msfe_fragment_classifier train --data-dir data\demo_fragments --out-dir outputs\demo_model --epochs 1 --batch-size 32
```

Run prediction on the demo test split:

```powershell
python -m msfe_fragment_classifier predict --model-dir outputs\demo_model --input-npz data\demo_fragments\test.npz --out outputs\demo_predictions.csv
```

The same commands are available through:

```powershell
.\scripts\train_demo.ps1
.\scripts\predict_demo.ps1
```

## Training on Your Dataset

The training folder can use either three fixed splits:

```text
dataset/
├── train.npz
├── val.npz
└── test.npz
```

or one combined file:

```text
dataset/
└── data.npz
```

Each NPZ file must contain a 2D byte array and labels. The feature array should have shape `[number_of_fragments, fragment_size]`; labels should be a 1D array of class ids.

```powershell
python -m msfe_fragment_classifier train --data-dir path\to\dataset --out-dir outputs\experiment_01 --epochs 30 --batch-size 128
```

## Prediction

Predict from an NPZ file:

```powershell
python -m msfe_fragment_classifier predict --model-dir outputs\experiment_01 --input-npz path\to\fragments.npz --out outputs\predictions.csv
```

Predict from a raw binary file:

```powershell
python -m msfe_fragment_classifier predict --model-dir outputs\experiment_01 --input-file path\to\sample.bin --out outputs\sample_predictions.csv
```

## Outputs

Training writes model files and metrics under the selected output directory:

```text
outputs/experiment_01/
├── model_metadata.json
├── base_models/
│   └── <variant_name>/
│       ├── best_model.keras
│       ├── summary.json
│       ├── confusion_matrix.csv
│       ├── per_class_metrics.csv
│       └── test_predictions.csv
└── stacker_hgb/
    ├── hgb_stacker.joblib
    ├── summary.json
    ├── confusion_matrix.csv
    ├── per_class_metrics.csv
    └── test_predictions.csv


