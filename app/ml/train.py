"""
train.py

Part of the GestureSpeak AI ML pipeline.

Trains a GestureMLP classifier on a CSV dataset of normalized landmark
feature vectors and saves the resulting model to app/ml/models/.

Expected CSV format (see generate_test_data.py for an example writer):
    label, synthetic, lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ..., lm20_z

  - `label`     : string class name (e.g. an ISL sign name)
  - `synthetic` : "True"/"False" -- whether the row is real ISL data
                  or synthetic placeholder data. This is carried
                  through into the saved model metadata so the app can
                  warn the user if it is only running on test data.
  - lmI_x/y/z   : the 63 raw (already 0-1 normalized-by-MediaPipe)
                  landmark coordinates for landmark index I.

NOTE: this script re-derives the ML feature vector from the raw
landmark columns via feature_extraction.extract_features(), so the
exact same normalization used at prediction time is guaranteed to be
used at training time.

Usage:
    # Train on the bundled synthetic TEST DATA (pipeline smoke test):
    python -m app.ml.generate_test_data
    python -m app.ml.train --data app/ml/data/synthetic_test_data.csv

    # Train on a real ISL dataset once one exists, in the same CSV
    # shape:
    python -m app.ml.train --data app/ml/data/isl_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.ml.feature_extraction import NUM_LANDMARKS, extract_features
from app.ml.model import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, GestureMLP


def load_dataset(csv_path: Path):
    """
    Reads the CSV dataset described in the module docstring and
    returns (X, y_indices, labels, all_synthetic).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Generate synthetic test data with "
            "`python -m app.ml.generate_test_data`, or provide a real "
            "ISL dataset CSV in the same format."
        )

    rows = []
    synthetic_flags = []
    labels_raw = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        coord_cols = [
            f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
        ]
        for row in reader:
            landmarks = [float(row[col]) for col in coord_cols]
            rows.append(extract_features(landmarks))
            labels_raw.append(row["label"])
            synthetic_flags.append(str(row.get("synthetic", "False")) == "True")

    if not rows:
        raise ValueError(f"Dataset at {csv_path} contains no rows.")

    X = np.stack(rows)
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
        help="Path to the training CSV (see module docstring for format).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    X, y_indices, labels, all_synthetic = load_dataset(args.data)
    print(f"Loaded {X.shape[0]} samples, {len(labels)} classes: {labels}")

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
    }
    model.save(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
    print(f"\nSaved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved metadata to {DEFAULT_METADATA_PATH}")

    if all_synthetic:
        print(
            "\nReminder: this model was trained on SYNTHETIC TEST DATA. "
            "Replace app/ml/data/synthetic_test_data.csv with a real, "
            "labeled ISL landmark dataset and re-run this script before "
            "relying on predictions."
        )


if __name__ == "__main__":
    main()
