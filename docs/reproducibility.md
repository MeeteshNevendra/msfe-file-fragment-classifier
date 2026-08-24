# Reproducibility Notes

The code sets a fixed random seed for Python, NumPy, and TensorFlow. Exact results can still vary slightly across TensorFlow versions, GPU drivers, CUDA builds, and CPU/GPU execution.

## Suggested Practice

- Record the Python version and package versions used for each run.
- Keep `model_metadata.json` together with trained model files.
- Use the same train, validation, and test files when comparing model variants.
- Avoid committing large datasets, model checkpoints, and run outputs.
- Store final result tables separately from temporary training logs.

## Output Files to Preserve

For a completed experiment, these files are usually enough for later inspection:

```text
model_metadata.json
stacker_hgb/summary.json
stacker_hgb/confusion_matrix.csv
stacker_hgb/per_class_metrics.csv
stacker_hgb/test_predictions.csv
```

Keep base-model summaries when ablation results are needed:

```text
base_models/<variant_name>/summary.json
base_models/<variant_name>/per_class_metrics.csv
base_models/<variant_name>/confusion_matrix.csv
```
