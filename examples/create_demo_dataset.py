import argparse
from pathlib import Path

import numpy as np


def make_fragment(rng, class_id, seq_len):
    mode = class_id % 5

    if mode == 0:
        fragment = np.zeros(seq_len, dtype=np.uint8)
        count = max(4, seq_len // 64)
        positions = rng.integers(0, seq_len, size=count)
        fragment[positions] = rng.integers(1, 48, size=count, dtype=np.uint8)
        return fragment

    if mode == 1:
        return rng.integers(32, 127, size=seq_len, dtype=np.uint8)

    if mode == 2:
        return rng.integers(0, 256, size=seq_len, dtype=np.uint8)

    if mode == 3:
        base = (np.arange(seq_len, dtype=np.uint16) + class_id * 17) % 256
        noise = rng.integers(0, 13, size=seq_len, dtype=np.uint16)
        return ((base + noise) % 256).astype(np.uint8)

    signature = np.array([(class_id * 29 + i * 11) % 256 for i in range(16)], dtype=np.uint8)
    repeats = int(np.ceil(seq_len / len(signature)))
    fragment = np.tile(signature, repeats)[:seq_len].copy()
    jitter = rng.random(seq_len) < 0.04
    fragment[jitter] = rng.integers(0, 256, size=int(jitter.sum()), dtype=np.uint8)
    return fragment


def build_split(rng, classes, samples_per_class, seq_len):
    rows = []
    labels = []
    for class_id in range(classes):
        for _ in range(samples_per_class):
            rows.append(make_fragment(rng, class_id, seq_len))
            labels.append(class_id)
    X = np.asarray(rows, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int32)
    order = rng.permutation(len(y))
    return X[order], y[order]


def save_three_way_dataset(out_dir, seq_len, classes, samples_per_class, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_n = max(20, int(samples_per_class * 0.70))
    val_n = max(8, int(samples_per_class * 0.15))
    test_n = max(8, samples_per_class - train_n - val_n)

    splits = {
        "train": build_split(rng, classes, train_n, seq_len),
        "val": build_split(rng, classes, val_n, seq_len),
        "test": build_split(rng, classes, test_n, seq_len),
    }

    for name, (X, y) in splits.items():
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)

    print(f"Saved demo dataset to {out_dir}")
    print(f"Classes: {classes}, fragment length: {seq_len}")
    print(f"Train/val/test per class: {train_n}/{val_n}/{test_n}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small byte-fragment dataset for a local smoke run.")
    parser.add_argument("--out-dir", default="data/demo_fragments", help="Output directory for train/val/test NPZ files.")
    parser.add_argument("--seq-len", type=int, default=512, help="Byte length of each fragment.")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes.")
    parser.add_argument("--samples-per-class", type=int, default=90, help="Approximate total samples per class before splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.classes < 2:
        raise ValueError("--classes must be at least 2")
    if args.seq_len < 64:
        raise ValueError("--seq-len must be at least 64")
    save_three_way_dataset(args.out_dir, args.seq_len, args.classes, args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
