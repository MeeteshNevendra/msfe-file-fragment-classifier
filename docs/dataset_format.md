# Dataset Format

The training command accepts either a three-split dataset or one combined dataset.

## Option 1: Three Splits

```text
dataset/
├── train.npz
├── val.npz
└── test.npz
```

Each file must contain:

- `X`: 2D byte-fragment array with shape `[samples, fragment_size]`
- `y`: 1D label array with shape `[samples]`

The loader also accepts these feature keys:

```text
X, x, data, features, feature, fragments, fragment, bytes, byte_data
```

and these label keys:

```text
y, Y, label, labels, target, targets, class, classes
```

## Option 2: Combined File

```text
dataset/
└── data.npz
```

When `data.npz` is used, the program creates stratified train, validation, and test splits in memory.

## Fragment Rules

- All fragments must have the same length.
- Fragment values should be integer byte values from 0 to 255.
- Labels should be class ids or sortable class labels.
- Training labels define the class order used in saved metadata and prediction output.

## Minimal Example

```python
import numpy as np

X = np.random.randint(0, 256, size=(1000, 512), dtype=np.uint8)
y = np.random.randint(0, 5, size=(1000,), dtype=np.int32)
np.savez_compressed("data.npz", X=X, y=y)
```
