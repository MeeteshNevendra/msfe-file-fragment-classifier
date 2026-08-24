#!/usr/bin/env python
"""
Standalone main code for the Proposed Multi-Scale Ensemble Framework.

This file contains only the core file-fragment classification method:
1. Load byte-fragment data.
2. Train three hard-coded multi-scale neural base learners.
3. Train an HGB meta-classifier over base-model probabilities.
4. Evaluate on a test split or predict labels for new fragments.

Expected training input:
    Option A:
        data_dir/train.npz
        data_dir/val.npz
        data_dir/test.npz

    Option B:
        data_dir/data.npz
        The script creates stratified train/val/test splits in memory.

Each NPZ must contain a 2D byte array X with shape [num_fragments, fragment_size]
and labels y for training/evaluation. Common key names are detected automatically:
X/x/data/features/bytes and y/Y/labels/target.

Example training:
    python proposed_file_fragment_classifier.py train --data-dir ./fft75_512_3 --out-dir ./trained_model

Example prediction from NPZ:
    python proposed_file_fragment_classifier.py predict --model-dir ./trained_model --input-npz ./new_fragments.npz --out predictions.csv

Example prediction from one binary file:
    python proposed_file_fragment_classifier.py predict --model-dir ./trained_model --input-file sample.bin --out predictions.csv
"""

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models


SEED = 42
X_KEYS = ["X", "x", "data", "features", "feature", "fragments", "fragment", "bytes", "byte_data"]
Y_KEYS = ["y", "Y", "label", "labels", "target", "targets", "class", "classes"]


BASE_VARIANTS = [
    {
        "name": "A_1_2_4_8_basic_singlehash",
        "ngrams": [1, 2, 4, 8],
        "enhanced_stats": False,
        "multihash": False,
        "hash_buckets": 65536,
    },
    {
        "name": "B_1_2_4_8_16_basic_singlehash",
        "ngrams": [1, 2, 4, 8, 16],
        "enhanced_stats": False,
        "multihash": False,
        "hash_buckets": 65536,
    },
    {
        "name": "C_1_2_4_8_16_enhanced_singlehash",
        "ngrams": [1, 2, 4, 8, 16],
        "enhanced_stats": True,
        "multihash": False,
        "hash_buckets": 65536,
    },
]


DEFAULT_TRAINING = {
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 2e-4,
    "weight_decay": 5e-5,
    "label_smoothing": 0.01,
    "focal_gamma": 1.8,
    "base_filters": 64,
    "embed_dim": 32,
    "early_stopping_patience": 6,
    "use_byte_dropout_aug": True,
    "byte_dropout_prob": 0.02,
    "gbflip": {
        "enabled": True,
        "prob": 0.003,
        "bit_center": 3.5,
        "bit_sigma": 1.6,
        "max_bits_per_byte": 1,
    },
}


HGB_PARAM_GRID = (
    {"learning_rate": 0.04, "max_iter": 180, "max_leaf_nodes": 15, "l2_regularization": 0.03},
    {"learning_rate": 0.05, "max_iter": 220, "max_leaf_nodes": 31, "l2_regularization": 0.03},
    {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.10},
    {"learning_rate": 0.06, "max_iter": 180, "max_leaf_nodes": 63, "l2_regularization": 0.10},
    {"learning_rate": 0.08, "max_iter": 140, "max_leaf_nodes": 31, "l2_regularization": 0.00},
)

BIAS_STEPS = (0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    seq_len: int
    num_classes: int
    class_names: list


def configure_runtime(seed=SEED, mixed_precision=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_key(npz, candidates, required=True):
    for key in candidates:
        if key in npz.files:
            return key
    if required:
        raise ValueError(f"Could not detect required key. Available keys: {npz.files}")
    return None


def load_npz_xy(path, require_y=True):
    npz = np.load(path, mmap_mode="r")
    x_key = detect_key(npz, X_KEYS, required=True)
    y_key = detect_key(npz, Y_KEYS, required=require_y)
    x = npz[x_key]
    y = None if y_key is None else np.asarray(npz[y_key])
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D byte-fragment array in {path}; got shape {x.shape}")
    return x, y


def remap_labels(y_train, y_val=None, y_test=None):
    class_values = np.unique(y_train)
    class_names = [str(v) for v in class_values.tolist()]

    def map_y(y):
        if y is None:
            return None
        mapped = np.searchsorted(class_values, y)
        valid = mapped < len(class_values)
        valid[valid] = class_values[mapped[valid]] == y[valid]
        if not np.all(valid):
            raise ValueError("Validation/test labels contain classes not present in training labels.")
        return mapped.astype(np.int32)

    return map_y(y_train), map_y(y_val), map_y(y_test), class_names


def load_three_split_dataset(data_dir):
    data_dir = Path(data_dir)
    x_train, y_train = load_npz_xy(data_dir / "train.npz", require_y=True)
    x_val, y_val = load_npz_xy(data_dir / "val.npz", require_y=True)
    x_test, y_test = load_npz_xy(data_dir / "test.npz", require_y=True)
    if x_val.shape[1] != x_train.shape[1] or x_test.shape[1] != x_train.shape[1]:
        raise ValueError("train/val/test fragment sizes do not match.")
    y_train, y_val, y_test, class_names = remap_labels(y_train, y_val, y_test)
    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        seq_len=int(x_train.shape[1]),
        num_classes=len(class_names),
        class_names=class_names,
    )


def load_single_npz_dataset(data_dir, val_size=0.1, test_size=0.1):
    data_dir = Path(data_dir)
    x, y_raw = load_npz_xy(data_dir / "data.npz", require_y=True)
    y_all, _unused, _unused2, class_names = remap_labels(y_raw)
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_val_idx, test_idx = next(splitter1.split(x, y_all))
    remaining_test_ratio = val_size / max(1e-8, 1.0 - test_size)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=remaining_test_ratio, random_state=SEED + 1)
    train_rel, val_rel = next(splitter2.split(x[train_val_idx], y_all[train_val_idx]))
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return SplitData(
        x_train=x[train_idx],
        y_train=y_all[train_idx],
        x_val=x[val_idx],
        y_val=y_all[val_idx],
        x_test=x[test_idx],
        y_test=y_all[test_idx],
        seq_len=int(x.shape[1]),
        num_classes=len(class_names),
        class_names=class_names,
    )


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    if (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists() and (data_dir / "test.npz").exists():
        return load_three_split_dataset(data_dir)
    if (data_dir / "data.npz").exists():
        return load_single_npz_dataset(data_dir)
    raise FileNotFoundError("Provide train.npz/val.npz/test.npz or data.npz in --data-dir.")


def encode_1byte(x):
    return np.asarray(x, dtype=np.uint8).astype(np.int32)


def encode_nbyte_hash(x, n, buckets, seed_prime=257, offset=1):
    x = np.asarray(x, dtype=np.uint64)
    usable = (x.shape[1] // n) * n
    x = x[:, :usable].reshape(x.shape[0], -1, n)
    h = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint64)
    prime = np.uint64(seed_prime)
    off = np.uint64(offset)
    for i in range(n):
        h = (h * prime + x[:, :, i] + off + np.uint64(i * 17)) % np.uint64(buckets)
    return h.astype(np.int32)


def compute_basic_stats(x):
    x_u8 = np.asarray(x, dtype=np.uint8)
    batch_size, seq_len = x_u8.shape
    hist = np.zeros((batch_size, 256), dtype=np.float32)
    for i in range(batch_size):
        hist[i] = np.bincount(x_u8[i], minlength=256).astype(np.float32)
    hist /= float(seq_len)
    mean = np.mean(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0
    std = np.std(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0
    entropy = -np.sum(hist * np.log2(hist + 1e-8), axis=1, keepdims=True).astype(np.float32) / 8.0
    zero_ratio = np.mean(x_u8 == 0, axis=1, keepdims=True).astype(np.float32)
    high_ratio = np.mean(x_u8 > 127, axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([hist, mean, std, entropy, zero_ratio, high_ratio], axis=1).astype(np.float32)


def longest_run(row, value=None):
    if len(row) == 0:
        return 0
    if value is None:
        best = cur = 1
        prev = row[0]
        for item in row[1:]:
            if item == prev:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
                prev = item
        return best
    best = cur = 0
    for item in row:
        if item == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_enhanced_stats(x):
    x_u8 = np.asarray(x, dtype=np.uint8)
    batch_size, seq_len = x_u8.shape
    hist = np.zeros((batch_size, 256), dtype=np.float32)
    for i in range(batch_size):
        hist[i] = np.bincount(x_u8[i], minlength=256).astype(np.float32)
    hist /= float(seq_len)

    diff = np.abs(np.diff(x_u8.astype(np.int16), axis=1)).astype(np.float32)
    sorted_hist = np.sort(hist, axis=1)
    q25 = np.percentile(x_u8, 25, axis=1, keepdims=True).astype(np.float32) / 255.0
    q75 = np.percentile(x_u8, 75, axis=1, keepdims=True).astype(np.float32) / 255.0
    stats = [
        hist,
        np.mean(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0,
        np.std(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0,
        -np.sum(hist * np.log2(hist + 1e-8), axis=1, keepdims=True).astype(np.float32) / 8.0,
        np.mean(x_u8 == 0, axis=1, keepdims=True).astype(np.float32),
        np.mean(x_u8 > 127, axis=1, keepdims=True).astype(np.float32),
        np.mean(x_u8 < 32, axis=1, keepdims=True).astype(np.float32),
        np.mean((x_u8 >= 32) & (x_u8 <= 126), axis=1, keepdims=True).astype(np.float32),
        np.mean((x_u8 < 32) | (x_u8 == 127), axis=1, keepdims=True).astype(np.float32),
        np.mean((x_u8 == 9) | (x_u8 == 10) | (x_u8 == 13) | (x_u8 == 32), axis=1, keepdims=True).astype(np.float32),
        np.min(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0,
        np.max(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0,
        np.median(x_u8, axis=1, keepdims=True).astype(np.float32) / 255.0,
        q25,
        q75,
        q75 - q25,
        np.array([len(np.unique(row)) for row in x_u8], dtype=np.float32).reshape(-1, 1) / 256.0,
        np.mean(diff, axis=1, keepdims=True) / 255.0,
        np.std(diff, axis=1, keepdims=True) / 255.0,
        np.mean(diff == 0, axis=1, keepdims=True).astype(np.float32),
        np.mean(diff > 127, axis=1, keepdims=True).astype(np.float32),
        np.array([longest_run(row, 0) for row in x_u8], dtype=np.float32).reshape(-1, 1) / float(seq_len),
        np.array([longest_run(row, None) for row in x_u8], dtype=np.float32).reshape(-1, 1) / float(seq_len),
        sorted_hist[:, -1].reshape(-1, 1).astype(np.float32),
        sorted_hist[:, -2].reshape(-1, 1).astype(np.float32),
        sorted_hist[:, -3].reshape(-1, 1).astype(np.float32),
        np.sum(hist ** 2, axis=1, keepdims=True).astype(np.float32),
    ]
    return np.concatenate(stats, axis=1).astype(np.float32)


def stats_dim(enhanced):
    dummy = np.zeros((2, 512), dtype=np.uint8)
    return compute_enhanced_stats(dummy).shape[1] if enhanced else compute_basic_stats(dummy).shape[1]


def build_model_inputs(x_batch, variant):
    buckets = int(variant["hash_buckets"])
    inputs = []
    for n in variant["ngrams"]:
        n = int(n)
        if n == 1:
            inputs.append(encode_1byte(x_batch))
        elif variant.get("multihash", False):
            inputs.append(encode_nbyte_hash(x_batch, n, buckets, seed_prime=257, offset=1))
            inputs.append(encode_nbyte_hash(x_batch, n, buckets, seed_prime=263, offset=29))
        else:
            inputs.append(encode_nbyte_hash(x_batch, n, buckets))
    if variant.get("enhanced_stats", False):
        inputs.append(compute_enhanced_stats(x_batch))
    else:
        inputs.append(compute_basic_stats(x_batch))
    return tuple(inputs)


def one_hot(y, num_classes, smoothing):
    encoded = tf.keras.utils.to_categorical(y.astype(np.int32), num_classes=num_classes).astype(np.float32)
    if smoothing > 0:
        encoded = encoded * (1.0 - smoothing) + smoothing / float(num_classes)
    return encoded


def apply_byte_dropout(x, prob, rng):
    if prob <= 0:
        return x
    out = np.array(x, copy=True)
    out[rng.random(out.shape) < prob] = 0
    return out


def apply_gbflip(x, prob, rng, bit_center=3.5, bit_sigma=1.6, max_bits_per_byte=1):
    if prob <= 0:
        return x
    out = np.array(x, copy=True).astype(np.uint8, copy=False)
    flip_mask = rng.random(out.shape) < prob
    rows, cols = np.where(flip_mask)
    if len(rows) == 0:
        return out
    for _ in range(max(int(max_bits_per_byte), 1)):
        bit_idx = np.rint(rng.normal(float(bit_center), float(bit_sigma), size=len(rows))).astype(np.int16)
        bit_idx = np.clip(bit_idx, 0, 7).astype(np.uint8)
        bit_mask = np.left_shift(np.uint8(1), bit_idx)
        out[rows, cols] = np.bitwise_xor(out[rows, cols], bit_mask)
    return out


def augment_batch(x, cfg, rng):
    out = x
    if bool(cfg.get("use_byte_dropout_aug", True)):
        out = apply_byte_dropout(out, float(cfg.get("byte_dropout_prob", 0.0)), rng)
    gbflip = cfg.get("gbflip", {})
    if bool(gbflip.get("enabled", False)):
        out = apply_gbflip(
            out,
            float(gbflip.get("prob", 0.0)),
            rng,
            bit_center=float(gbflip.get("bit_center", 3.5)),
            bit_sigma=float(gbflip.get("bit_sigma", 1.6)),
            max_bits_per_byte=int(gbflip.get("max_bits_per_byte", 1)),
        )
    return out


class BalancedSampler:
    def __init__(self, y, sampling_weights=None):
        self.y = y.astype(np.int32)
        self.rng = np.random.default_rng(SEED)
        self.classes = np.unique(self.y)
        self.class_to_idx = {int(cls): np.where(self.y == cls)[0] for cls in self.classes}
        if sampling_weights is None:
            probs = np.ones(len(self.classes), dtype=np.float64)
        else:
            probs = np.array([sampling_weights[int(cls)] for cls in self.classes], dtype=np.float64)
        self.probs = probs / probs.sum()

    def sample(self, batch_size):
        classes = self.rng.choice(self.classes, size=batch_size, replace=True, p=self.probs)
        idx = [int(self.rng.choice(self.class_to_idx[int(cls)])) for cls in classes]
        self.rng.shuffle(idx)
        return np.asarray(idx, dtype=np.int64)


def data_generator(x, y, variant, cfg, seq_len, num_classes, sampling_weights=None, shuffle=True, augment=False):
    batch_size = int(cfg["batch_size"])
    smoothing = float(cfg.get("label_smoothing", 0.0))
    rng = np.random.default_rng(SEED)
    sampler = BalancedSampler(y, sampling_weights) if sampling_weights is not None else None
    while True:
        if sampler is not None:
            for _ in range(math.ceil(len(x) / batch_size)):
                idx = sampler.sample(batch_size)
                x_batch = np.asarray(x[idx])
                if augment:
                    x_batch = augment_batch(x_batch, cfg, rng)
                yield build_model_inputs(x_batch, variant), one_hot(y[idx], num_classes, smoothing)
        else:
            idx = np.arange(len(x))
            if shuffle:
                rng.shuffle(idx)
            for start in range(0, len(x), batch_size):
                batch_idx = idx[start:start + batch_size]
                x_batch = np.asarray(x[batch_idx])
                if augment:
                    x_batch = augment_batch(x_batch, cfg, rng)
                yield build_model_inputs(x_batch, variant), one_hot(y[batch_idx], num_classes, smoothing)


def output_signature(seq_len, num_classes, variant):
    specs = []
    for n in variant["ngrams"]:
        n = int(n)
        if n == 1:
            specs.append(tf.TensorSpec(shape=(None, seq_len), dtype=tf.int32))
        else:
            count = 2 if variant.get("multihash", False) else 1
            for _ in range(count):
                specs.append(tf.TensorSpec(shape=(None, seq_len // n), dtype=tf.int32))
    specs.append(tf.TensorSpec(shape=(None, stats_dim(variant.get("enhanced_stats", False))), dtype=tf.float32))
    return tuple(specs), tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)


def make_tf_dataset(x, y, variant, cfg, seq_len, num_classes, sampling_weights=None, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(x, y, variant, cfg, seq_len, num_classes, sampling_weights, shuffle, augment),
        output_signature=output_signature(seq_len, num_classes, variant),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, gamma=1.5):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="weighted_focal_loss")
        self.class_weights = tf.constant(np.asarray(class_weights, dtype=np.float32), dtype=tf.float32)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, self.gamma)
        weights = tf.reduce_sum(y_true * self.class_weights, axis=-1)
        return tf.reduce_sum(ce * focal, axis=-1) * weights


def conv_block(x, filters, kernel_size, weight_decay, dropout):
    x = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def se_block(x, ratio=8):
    channels = int(x.shape[-1])
    gate = layers.GlobalAveragePooling1D()(x)
    gate = layers.Dense(max(channels // ratio, 8), activation="relu")(gate)
    gate = layers.Dense(channels, activation="sigmoid")(gate)
    gate = layers.Reshape((1, channels))(gate)
    return layers.Multiply()([x, gate])


def sequence_branch(inp, vocab_size, cfg, name):
    embed_dim = int(cfg.get("embed_dim", 32))
    base_filters = int(cfg.get("base_filters", 64))
    weight_decay = float(cfg.get("weight_decay", 5e-5))
    x = layers.Embedding(vocab_size, embed_dim, name=f"{name}_embedding")(inp)
    x = conv_block(x, base_filters, 7, weight_decay, 0.05)
    x = layers.MaxPooling1D(2)(x)
    x = conv_block(x, base_filters, 5, weight_decay, 0.10)
    x = conv_block(x, base_filters * 2, 3, weight_decay, 0.10)
    x = se_block(x)
    return layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])


def stats_branch(inp):
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.10)(x)
    return layers.BatchNormalization()(x)


def build_base_model(seq_len, num_classes, variant, cfg):
    inputs = []
    branches = []
    for n in variant["ngrams"]:
        n = int(n)
        if n == 1:
            inp = layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_1byte")
            inputs.append(inp)
            branches.append(sequence_branch(inp, 256, cfg, "b1"))
        else:
            count = 2 if variant.get("multihash", False) else 1
            for hash_id in range(count):
                inp = layers.Input(shape=(seq_len // n,), dtype=tf.int32, name=f"input_{n}byte_h{hash_id + 1}")
                inputs.append(inp)
                branches.append(sequence_branch(inp, int(variant["hash_buckets"]), cfg, f"b{n}_{hash_id + 1}"))

    stats_input = layers.Input(shape=(stats_dim(variant.get("enhanced_stats", False)),), dtype=tf.float32, name="input_stats")
    inputs.append(stats_input)
    branches.append(stats_branch(stats_input))

    weight_decay = float(cfg.get("weight_decay", 5e-5))
    fused = layers.Concatenate()(branches)
    fused = layers.BatchNormalization()(fused)
    gate = layers.Dense(max(int(fused.shape[-1]) // 4, 32), activation="relu")(fused)
    gate = layers.Dense(int(fused.shape[-1]), activation="sigmoid")(gate)
    fused = layers.Multiply()([fused, gate])
    fused = layers.Dense(256, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.25)(fused)
    fused = layers.Dense(128, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.20)(fused)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(fused)
    return models.Model(inputs=inputs, outputs=out, name=f"base_{variant['name']}")


def predict_stream(model, x, variant, batch_size):
    probs = []
    for start in range(0, len(x), batch_size):
        x_batch = np.asarray(x[start:start + batch_size])
        probs.append(model.predict(build_model_inputs(x_batch, variant), verbose=0))
        gc.collect()
    return np.vstack(probs).astype(np.float32)


def evaluate_predictions(y_true, probs, class_names, forced_preds=None):
    num_classes = len(class_names)
    preds = np.asarray(forced_preds, dtype=np.int32) if forced_preds is not None else np.argmax(probs, axis=1).astype(np.int32)
    report = classification_report(y_true, preds, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    summary = {
        "accuracy_percent": round(accuracy_score(y_true, preds) * 100.0, 4),
        "top3_accuracy_percent": None,
        "top5_accuracy_percent": None,
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }
    if num_classes > 2:
        summary["top3_accuracy_percent"] = round(top_k_accuracy_score(y_true, probs, k=min(3, num_classes), labels=list(range(num_classes))) * 100.0, 4)
        summary["top5_accuracy_percent"] = round(top_k_accuracy_score(y_true, probs, k=min(5, num_classes), labels=list(range(num_classes))) * 100.0, 4)
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
    rows = []
    for cls in range(num_classes):
        mask = y_true == cls
        support = int(mask.sum())
        correct = int((preds[mask] == cls).sum()) if support else 0
        rows.append(
            {
                "class_id": cls,
                "class_name": class_names[cls],
                "support": support,
                "correct": correct,
                "class_accuracy_percent": round((correct / support * 100.0) if support else 0.0, 4),
                "precision": round(report.get(str(cls), {}).get("precision", 0.0), 4),
                "recall": round(report.get(str(cls), {}).get("recall", 0.0), 4),
                "f1_score": round(report.get(str(cls), {}).get("f1-score", 0.0), 4),
            }
        )
    return preds, summary, cm, pd.DataFrame(rows)


def train_one_base_model(ds, variant, out_dir, cfg, force=False):
    variant_dir = Path(out_dir) / "base_models" / variant["name"]
    model_path = variant_dir / "best_model.keras"
    summary_path = variant_dir / "summary.json"
    if model_path.exists() and summary_path.exists() and not force:
        print(f"Skipping existing model: {model_path}")
        return model_path

    variant_dir.mkdir(parents=True, exist_ok=True)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(ds.num_classes), y=ds.y_train).astype(np.float32)
    class_weights = np.power(class_weights, 0.5)
    class_weights = class_weights / np.mean(class_weights)
    sampling_weights = class_weights / np.mean(class_weights)

    train_ds = make_tf_dataset(ds.x_train, ds.y_train, variant, cfg, ds.seq_len, ds.num_classes, sampling_weights, shuffle=True, augment=True)
    val_ds = make_tf_dataset(ds.x_val, ds.y_val, variant, cfg, ds.seq_len, ds.num_classes, None, shuffle=False, augment=False)

    model = build_base_model(ds.seq_len, ds.num_classes, variant, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"]), clipnorm=1.0),
        loss=WeightedFocalLoss(class_weights, gamma=float(cfg.get("focal_gamma", 1.5))),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    start_time = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["epochs"]),
        steps_per_epoch=math.ceil(len(ds.x_train) / int(cfg["batch_size"])),
        validation_steps=math.ceil(len(ds.x_val) / int(cfg["batch_size"])),
        callbacks=[
            callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
            callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=int(cfg["early_stopping_patience"]), restore_best_weights=True, verbose=1),
            callbacks.CSVLogger(str(variant_dir / "training_log.csv")),
            callbacks.TerminateOnNaN(),
        ],
        verbose=1,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = predict_stream(model, ds.x_test, variant, int(cfg["batch_size"]))
    preds, summary, cm, per_class = evaluate_predictions(ds.y_test, probs, ds.class_names)
    summary.update(
        {
            "variant": variant,
            "seq_len": ds.seq_len,
            "num_classes": ds.num_classes,
            "training_time_seconds": round(time.time() - start_time, 2),
        }
    )
    save_json(summary_path, summary)
    pd.DataFrame(cm).to_csv(variant_dir / "confusion_matrix.csv", index=False)
    per_class.to_csv(variant_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame({"y_true": ds.y_test, "y_pred": preds, "confidence": np.max(probs, axis=1)}).to_csv(variant_dir / "test_predictions.csv", index=False)
    print(f"Saved base-model summary: {summary_path}")
    return model_path


def stable_log_probs(probs):
    return np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)


def softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def confidence_features(probs):
    probs = np.clip(probs.astype(np.float32), 1e-8, 1.0)
    num_classes = probs.shape[1]
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if num_classes > 1 else np.zeros_like(top1)
    top3_mass = np.sum(sorted_probs[:, -min(3, num_classes):], axis=1)
    entropy = -np.sum(probs * np.log(probs), axis=1) / max(np.log(float(num_classes)), 1e-8)
    margin = top1 - top2
    return np.stack([top1, top2, margin, top3_mass, entropy], axis=1).astype(np.float32)


def agreement_features(prob_list):
    probs = [np.clip(p.astype(np.float32), 1e-8, 1.0) for p in prob_list]
    stacked = np.stack(probs, axis=0)
    preds = np.stack([np.argmax(p, axis=1).astype(np.int32) for p in probs], axis=1)
    confidences = np.stack([np.max(p, axis=1) for p in probs], axis=1)
    margins = np.stack([confidence_features(p)[:, 2] for p in probs], axis=1)
    entropies = np.stack([confidence_features(p)[:, 4] for p in probs], axis=1)
    num_models = len(prob_list)
    num_classes = probs[0].shape[1]
    vote_strength = np.zeros(preds.shape[0], dtype=np.float32)
    for i, row in enumerate(preds):
        vote_strength[i] = np.bincount(row, minlength=num_classes).max() / float(num_models)
    mean_probs = np.mean(stacked, axis=0)
    return np.concatenate(
        [
            confidence_features(mean_probs),
            np.mean(confidences, axis=1, keepdims=True),
            np.std(confidences, axis=1, keepdims=True),
            np.min(confidences, axis=1, keepdims=True),
            np.max(confidences, axis=1, keepdims=True),
            np.mean(margins, axis=1, keepdims=True),
            np.std(margins, axis=1, keepdims=True),
            np.mean(entropies, axis=1, keepdims=True),
            np.std(entropies, axis=1, keepdims=True),
            vote_strength.reshape(-1, 1),
            (vote_strength == 1.0).astype(np.float32).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def stack_features(prob_list):
    feats = [stable_log_probs(p) for p in prob_list] + [p.astype(np.float32) for p in prob_list]
    for probs in prob_list:
        feats.append(confidence_features(probs))
    feats.append(agreement_features(prob_list))
    for i in range(len(prob_list)):
        for j in range(i + 1, len(prob_list)):
            feats.append((prob_list[i] - prob_list[j]).astype(np.float32))
            feats.append(np.abs(prob_list[i] - prob_list[j]).astype(np.float32))
            feats.append((prob_list[i] * prob_list[j]).astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)


def tune_class_bias(scores, y_true):
    num_classes = scores.shape[1]
    bias = np.zeros(num_classes, dtype=np.float32)
    best_acc = accuracy_score(y_true, np.argmax(scores, axis=1).astype(np.int32))
    for step in BIAS_STEPS:
        improved = True
        passes = 0
        while improved and passes < 20:
            improved = False
            passes += 1
            for cls in range(num_classes):
                current = float(bias[cls])
                local_acc = best_acc
                local_bias = current
                for candidate in (current - step, current, current + step):
                    bias[cls] = candidate
                    preds = np.argmax(scores + bias.reshape(1, -1), axis=1).astype(np.int32)
                    acc = accuracy_score(y_true, preds)
                    if acc > local_acc + 1e-12:
                        local_acc = acc
                        local_bias = candidate
                bias[cls] = local_bias
                if local_acc > best_acc + 1e-12:
                    best_acc = local_acc
                    improved = True
        bias -= np.mean(bias)
    return bias.astype(np.float32), float(best_acc)


def load_base_model_probs(model_dir, x, cfg):
    probs = []
    for variant in BASE_VARIANTS:
        model_path = Path(model_dir) / "base_models" / variant["name"] / "best_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained base model: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        probs.append(predict_stream(model, x, variant, int(cfg["batch_size"])))
        del model
        gc.collect()
    return probs


def train_stacker(ds, out_dir, cfg):
    out_dir = Path(out_dir)
    stacker_dir = out_dir / "stacker_hgb"
    stacker_dir.mkdir(parents=True, exist_ok=True)
    print("Predicting validation/test probabilities for stacker.")
    val_probs = load_base_model_probs(out_dir, ds.x_val, cfg)
    test_probs = load_base_model_probs(out_dir, ds.x_test, cfg)
    x_val_stack = stack_features(val_probs)
    x_test_stack = stack_features(test_probs)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 17)
    train_idx, holdout_idx = next(splitter.split(x_val_stack, ds.y_val))
    best = None
    for params in HGB_PARAM_GRID:
        model = HistGradientBoostingClassifier(**params, random_state=SEED, early_stopping=True, validation_fraction=0.12, n_iter_no_change=20)
        model.fit(x_val_stack[train_idx], ds.y_val[train_idx])
        scores = stable_log_probs(model.predict_proba(x_val_stack[holdout_idx]).astype(np.float32))
        raw_acc = accuracy_score(ds.y_val[holdout_idx], np.argmax(scores, axis=1).astype(np.int32))
        bias, tuned_acc = tune_class_bias(scores, ds.y_val[holdout_idx])
        print(f"HGB {params}: raw={raw_acc * 100.0:.4f}, calibrated={tuned_acc * 100.0:.4f}")
        row = {"params": params, "bias": bias, "holdout_accuracy": tuned_acc}
        if best is None or tuned_acc > best["holdout_accuracy"]:
            best = row

    final_model = HistGradientBoostingClassifier(**best["params"], random_state=SEED, early_stopping=True, validation_fraction=0.12, n_iter_no_change=20)
    final_model.fit(x_val_stack, ds.y_val)
    test_scores = stable_log_probs(final_model.predict_proba(x_test_stack).astype(np.float32)) + best["bias"].reshape(1, -1)
    test_probs = softmax(test_scores)
    preds, summary, cm, per_class = evaluate_predictions(ds.y_test, test_probs, ds.class_names)
    summary.update(
        {
            "method": "A+B+C neural ensemble with HGB meta-classifier",
            "seq_len": ds.seq_len,
            "num_classes": ds.num_classes,
            "stacker_features": int(x_test_stack.shape[1]),
            "hgb_params": best["params"],
            "hgb_holdout_accuracy_percent": round(best["holdout_accuracy"] * 100.0, 4),
            "class_bias": [round(float(v), 8) for v in best["bias"]],
        }
    )
    joblib.dump(final_model, stacker_dir / "hgb_stacker.joblib")
    save_json(stacker_dir / "summary.json", summary)
    pd.DataFrame(cm).to_csv(stacker_dir / "confusion_matrix.csv", index=False)
    per_class.to_csv(stacker_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame({"y_true": ds.y_test, "y_pred": preds, "confidence": np.max(test_probs, axis=1)}).to_csv(stacker_dir / "test_predictions.csv", index=False)
    print(f"Final ensemble accuracy: {summary['accuracy_percent']:.4f}%")
    return summary


def save_metadata(out_dir, ds, cfg):
    save_json(
        Path(out_dir) / "model_metadata.json",
        {
            "framework": "Proposed Multi-Scale Ensemble Framework",
            "seq_len": ds.seq_len,
            "num_classes": ds.num_classes,
            "class_names": ds.class_names,
            "base_variants": BASE_VARIANTS,
            "training": cfg,
        },
    )


def train_framework(args):
    cfg = dict(DEFAULT_TRAINING)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    configure_runtime(mixed_precision=args.mixed_precision)
    ds = load_dataset(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(out_dir, ds, cfg)

    print(f"Loaded dataset: seq_len={ds.seq_len}, classes={ds.num_classes}")
    for variant in BASE_VARIANTS:
        print(f"\nTraining base learner: {variant['name']}")
        train_one_base_model(ds, variant, out_dir, cfg, force=args.force)
    print("\nTraining ensemble stacker.")
    train_stacker(ds, out_dir, cfg)


def load_prediction_npz(path):
    x, y = load_npz_xy(path, require_y=False)
    return np.asarray(x), y


def load_binary_fragments(path, seq_len, pad_final=True):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if len(data) == 0:
        raise ValueError(f"Input file is empty: {path}")
    remainder = len(data) % seq_len
    if remainder:
        if pad_final:
            pad = seq_len - remainder
            data = np.pad(data, (0, pad), constant_values=0)
        else:
            data = data[: len(data) - remainder]
    return data.reshape(-1, seq_len)


def predict_with_framework(model_dir, x):
    metadata = load_json(Path(model_dir) / "model_metadata.json")
    cfg = metadata["training"]
    seq_len = int(metadata["seq_len"])
    if x.ndim != 2 or x.shape[1] != seq_len:
        raise ValueError(f"Expected input fragments with shape [N, {seq_len}], got {x.shape}")
    base_probs = load_base_model_probs(model_dir, x, cfg)
    x_stack = stack_features(base_probs)
    stacker = joblib.load(Path(model_dir) / "stacker_hgb" / "hgb_stacker.joblib")
    summary = load_json(Path(model_dir) / "stacker_hgb" / "summary.json")
    bias = np.asarray(summary["class_bias"], dtype=np.float32)
    scores = stable_log_probs(stacker.predict_proba(x_stack).astype(np.float32)) + bias.reshape(1, -1)
    probs = softmax(scores)
    preds = np.argmax(probs, axis=1).astype(np.int32)
    conf = np.max(probs, axis=1)
    class_names = metadata["class_names"]
    return preds, conf, probs, class_names


def predict_command(args):
    metadata = load_json(Path(args.model_dir) / "model_metadata.json")
    if args.input_npz:
        x, y_raw = load_prediction_npz(args.input_npz)
    elif args.input_file:
        x = load_binary_fragments(args.input_file, int(metadata["seq_len"]), pad_final=not args.drop_last)
        y_raw = None
    else:
        raise ValueError("Provide --input-npz or --input-file.")

    preds, conf, probs, class_names = predict_with_framework(args.model_dir, x)
    rows = {
        "fragment_index": np.arange(len(preds)),
        "predicted_class_id": preds,
        "predicted_class_name": [class_names[int(p)] for p in preds],
        "confidence": conf,
    }
    if y_raw is not None:
        rows["true_label"] = y_raw
    output = pd.DataFrame(rows)
    for class_id, class_name in enumerate(class_names):
        output[f"prob_{class_id}_{class_name}"] = probs[:, class_id]
    output.to_csv(args.out, index=False)
    print(f"Saved predictions: {args.out}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Proposed file-fragment classification framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train base learners and HGB ensemble.")
    train.add_argument("--data-dir", required=True, help="Folder with train/val/test.npz or data.npz.")
    train.add_argument("--out-dir", required=True, help="Output folder for trained models and metrics.")
    train.add_argument("--epochs", type=int, default=None, help="Override default epochs.")
    train.add_argument("--batch-size", type=int, default=None, help="Override default batch size.")
    train.add_argument("--force", action="store_true", help="Retrain even if model outputs already exist.")
    train.add_argument("--mixed-precision", action="store_true", help="Use TensorFlow mixed precision.")

    predict = sub.add_parser("predict", help="Predict labels for new fragments using a trained framework.")
    predict.add_argument("--model-dir", required=True, help="Folder created by the train command.")
    group = predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-npz", help="NPZ file containing a 2D byte-fragment array.")
    group.add_argument("--input-file", help="Raw binary file to split into fragments.")
    predict.add_argument("--drop-last", action="store_true", help="Drop incomplete final raw-file fragment instead of zero-padding.")
    predict.add_argument("--out", default="predictions.csv", help="CSV output path.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.command == "train":
        train_framework(args)
    elif args.command == "predict":
        configure_runtime()
        predict_command(args)


if __name__ == "__main__":
    main()
