"""
train.py

Part of the GestureSpeak AI ML pipeline.

Trains a GestureMLP classifier on a CSV dataset of landmark examples
and saves the resulting model to app/ml/models/.

TWO supported CSV formats (auto-detected from the header):

  1. LEGACY flat format (e.g. app/ml/data/synthetic_test_data.csv,
     written by generate_test_data.py):
         label, synthetic, lm0_x, lm0_y, lm0_z, ..., lm20_z
     Each row is treated as one independent STATIC example
     (frame_count=1).

  2. Stage 2 sequence-capable format (written by collect_data.py,
     e.g. app/ml/data/isl_real_data_v2.csv):
         example_id, label, type, frame_index, frame_count,
         synthetic, lm0_x, lm0_y, lm0_z, ..., lm20_z
     Rows sharing the same `example_id` are ONE example. `type` is
     "static" (always exactly 1 row) or "dynamic" (frame_count rows,
     ordered by `frame_index`).

STAGE 3 CORRECTION: every example -- static or dynamic -- used to be
converted into a 20-frame flattened window (repeating a static frame
20x, or resampling a dynamic sequence to 20 frames -> 1260 features).
Diagnostics proved that design destabilized training: tiling one
frame into 20 IDENTICAL copies made backprop apply the same gradient
to 20 weight blocks every step, equivalent to a ~20x-too-high
effective learning rate on the first Dense layer (confirmed: dividing
the learning rate by 20 restored 100% accuracy on the exact same data;
adding more hidden capacity at the original learning rate made it
worse, ruling out a capacity explanation).

Regardless of format, every example is now converted by
app.ml.sequence_utils.build_compact_features() into a single 189-dim
feature vector: [pose (63), delta (63), summary (63)], where pose is
the first frame's normalized features, delta is last-minus-first
frame, and summary is mean-minus-first frame. A static example (1
frame) naturally produces delta=0 and summary=0 -- not duplicated
data, so it doesn't reproduce the old tiling pathology -- while a
dynamic example's delta/summary carry its real motion. This lets
static and dynamic examples train together in one GestureMLP without
any special-casing or learning-rate workaround.
`extract_features()` (unchanged) still performs the actual per-frame
normalization; sequence_utils only handles the temporal side.

Safety rule: a single training run must not mix synthetic placeholder
data with real ISL data. If a dataset file contains a mix of
synthetic=True and synthetic=False examples, load_dataset() raises an
error rather than silently training on a blended dataset. Point
--data at ONE file (the synthetic test file, or a real ISL file) per
run.

Usage:
    # Train on the bundled synthetic TEST DATA (pipeline smoke test):
    python -m app.ml.generate_test_data
    python -m app.ml.train --data app/ml/data/synthetic_test_data.csv

    # Train on real, collected ISL data (Stage 2 collector output),
    # with static and dynamic examples mixed together in one file:
    python -m app.ml.train --data app/ml/data/isl_real_data_v2.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.ml.model import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, GestureMLP
from app.ml.sequence_utils import (
    build_compact_features,
    group_rows_by_example,
)

LEGACY_FORMAT_COLUMNS = {"label", "synthetic"}
SEQUENCE_FORMAT_COLUMNS = {"example_id", "label", "type", "frame_index", "frame_count", "synthetic"}


def _coord_cols_from_fieldnames(fieldnames) -> list:
    """
    Recover the ordered list of lmI_axis columns from whatever header
    is present, rather than assuming NUM_LANDMARKS, so this keeps
    working even if the landmark count ever changes.
    """
    coord_cols = [c for c in fieldnames if c.startswith("lm")]
    return coord_cols


def _load_legacy_format(reader: csv.DictReader):
    """
    Each row = one independent STATIC example (frame_count=1).
    Returns a list of (label, synthetic_bool, compact_feature_vector)
    tuples.
    """
    coord_cols = _coord_cols_from_fieldnames(reader.fieldnames)
    examples = []
    for row in reader:
        raw_frame = [float(row[col]) for col in coord_cols]
        features = build_compact_features([raw_frame], "static")
        synthetic = str(row.get("synthetic", "False")) == "True"
        examples.append((row["label"], synthetic, features))
    return examples


def _load_sequence_format(reader: csv.DictReader):
    """
    Groups rows by example_id. Static groups have exactly 1 row;
    dynamic groups have frame_count rows sorted by frame_index.
    Returns a list of (label, synthetic_bool, compact_feature_vector)
    tuples, one per example (not per row).
    """
    coord_cols = _coord_cols_from_fieldnames(reader.fieldnames)
    all_rows = list(reader)
    groups = group_rows_by_example(all_rows)

    examples = []
    for example_id, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda r: int(r["frame_index"]))

        types = {r["type"].strip().lower() for r in group_rows}
        if len(types) != 1:
            raise ValueError(
                f"Example '{example_id}' has inconsistent 'type' values "
                f"across its rows: {types}. Each example_id must be "
                "entirely static or entirely dynamic."
            )
        example_type = types.pop()

        synthetic_flags = {str(r.get("synthetic", "False")) == "True" for r in group_rows}
        if len(synthetic_flags) != 1:
            raise ValueError(
                f"Example '{example_id}' has inconsistent 'synthetic' "
                "values across its rows."
            )
        synthetic = synthetic_flags.pop()

        expected_frame_count = int(group_rows[0]["frame_count"])
        if len(group_rows) != expected_frame_count:
            print(
                f"Warning: example '{example_id}' declares "
                f"frame_count={expected_frame_count} but has "
                f"{len(group_rows)} rows -- using the rows actually "
                "present."
            )

        raw_frames = [
            [float(r[col]) for col in coord_cols] for r in group_rows
        ]
        features = build_compact_features(raw_frames, example_type)
        label = group_rows[0]["label"]
        examples.append((label, synthetic, features))

    return examples


def load_dataset(csv_path: Path):
    """
    Reads either supported CSV format (auto-detected) and returns
    (X, y_indices, labels, all_synthetic).

    X has shape (num_examples, 189) -- one row per EXAMPLE (not per
    captured frame): a static example contributes exactly one row of
    X, and a dynamic example's whole sequence is collapsed into
    exactly one row of X via
    sequence_utils.build_compact_features().
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Generate synthetic test data with "
            "`python -m app.ml.generate_test_data`, or collect a real "
            "ISL dataset with `python -m app.ml.collect_data`."
        )

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        if SEQUENCE_FORMAT_COLUMNS.issubset(fieldnames):
            examples = _load_sequence_format(reader)
        elif LEGACY_FORMAT_COLUMNS.issubset(fieldnames):
            examples = _load_legacy_format(reader)
        else:
            raise ValueError(
                f"Unrecognized CSV format in {csv_path}. Expected either "
                f"the legacy columns {sorted(LEGACY_FORMAT_COLUMNS)} or the "
                f"sequence-format columns {sorted(SEQUENCE_FORMAT_COLUMNS)}."
            )

    if not examples:
        raise ValueError(f"Dataset at {csv_path} contains no examples.")

    labels_raw = [ex[0] for ex in examples]
    synthetic_flags = [ex[1] for ex in examples]
    X = np.stack([ex[2] for ex in examples])

    # Safety rule: never silently blend synthetic placeholder data
    # with real ISL data in one training run.
    if any(synthetic_flags) and not all(synthetic_flags):
        num_synthetic = sum(synthetic_flags)
        num_real = len(synthetic_flags) - num_synthetic
        raise ValueError(
            f"Dataset at {csv_path} mixes {num_synthetic} synthetic "
            f"example(s) with {num_real} real example(s) in the same "
            "file. Keep synthetic test data and real ISL data in "
            "separate files and train on one at a time."
        )

    labels = sorted(set(labels_raw))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_indices = np.array([label_to_idx[label] for label in labels_raw])
    all_synthetic = all(synthetic_flags)

    return X, y_indices, labels, all_synthetic


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    parser = argparse.ArgumentParser(description="Train the GestureSpeak MLP model.")
    parser.add_argument(
        "--data",
        type=str,
        default="app/ml/data/synthetic_test_data.csv",
        help="Path to the training CSV (see module docstring for supported formats).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    X, y_indices, labels, all_synthetic = load_dataset(args.data)
    print(f"Loaded {X.shape[0]} examples, {len(labels)} classes: {labels}")
    print(f"Compact feature dimension: {X.shape[1]} (= pose[63] + delta[63] + summary[63])")

    if all_synthetic:
        print(
            "\n*** WARNING: every row in this dataset is marked "
            "synthetic=True. ***\n"
            "*** This model is a TECHNICAL PIPELINE TEST ONLY and "
            "does NOT recognize real ISL gestures. ***\n"
        )

    X_train, y_train, X_test, y_test = train_test_split(X, y_indices)

    model = GestureMLP(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        labels=labels,
    )

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    test_preds, _ = model.predict(X_test)
    test_labels = [labels[i] for i in y_test]
    test_acc = float(np.mean([p == t for p, t in zip(test_preds, test_labels)]))
    print(f"\nHeld-out test accuracy: {test_acc:.3f} ({len(y_test)} samples)")

    model.metadata = {
        "data_source": str(args.data),
        "is_synthetic_test_data": all_synthetic,
        "num_training_samples": int(X_train.shape[0]),
        "num_test_samples": int(X_test.shape[0]),
        "test_accuracy": test_acc,
        "feature_representation": "compact_pose_delta_summary_v1",
        "feature_dim": int(X.shape[1]),
    }
    model.save(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
    print(f"\nSaved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved metadata to {DEFAULT_METADATA_PATH}")

    if all_synthetic:
        print(
            "\nReminder: this model was trained on SYNTHETIC TEST DATA. "
            "Collect real ISL data with `python -m app.ml.collect_data` "
            "and re-run this script against that file before relying on "
            "predictions."
        )


if __name__ == "__main__":
    main()
