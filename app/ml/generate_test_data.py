"""
generate_test_data.py

*** THIS PRODUCES SYNTHETIC TEST DATA ONLY. IT IS NOT REAL ISL DATA. ***

GestureSpeak AI does not yet have a real Indian Sign Language landmark
dataset. This script exists solely so the ML pipeline (feature
extraction -> model -> train -> save -> load -> predict -> Flask ->
browser) can be exercised end-to-end and verified to work technically.

Every label produced by this script is prefixed with "TEST_" and every
row is written with a `synthetic=True` flag so it can never be
confused with, or accidentally shipped as, real ISL training data.

DO NOT use the output of this script to claim the app "recognizes ISL
signs". It only proves the plumbing works.

Usage:
    python -m app.ml.generate_test_data
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.ml.feature_extraction import FEATURE_LENGTH, NUM_LANDMARKS

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "synthetic_test_data.csv"

# Purely synthetic placeholder classes -- NOT real ISL signs.
TEST_CLASSES = ["TEST_OPEN_HAND", "TEST_CLOSED_FIST", "TEST_POINT_UP"]

SAMPLES_PER_CLASS = 150


def _base_hand_shape(shape_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Build a plausible, but entirely synthetic, (21, 3) hand-landmark
    "shape" for a given TEST class. These do not correspond to any
    real gesture; they are just three distinguishable geometric
    patterns so a classifier has something structurally different to
    learn to separate.
    """
    points = np.zeros((NUM_LANDMARKS, 3))
    # wrist at origin
    points[0] = [0.5, 0.9, 0.0]

    # crude finger chains (4 points each) fanning out from the wrist,
    # with a different fan angle/spread per synthetic class so the
    # three classes are geometrically separable.
    finger_bases = [1, 5, 9, 13, 17]  # thumb, index, middle, ring, pinky starts
    finger_lengths = [3, 3, 4, 4, 4]  # approx joints per finger in MediaPipe layout

    if shape_name == "TEST_OPEN_HAND":
        spread = np.linspace(-0.35, 0.35, len(finger_bases))
        extension = 1.0
    elif shape_name == "TEST_CLOSED_FIST":
        spread = np.linspace(-0.12, 0.12, len(finger_bases))
        extension = 0.25
    else:  # TEST_POINT_UP
        spread = np.linspace(-0.3, 0.3, len(finger_bases))
        extension = 1.0

    idx = 1
    for f, (base_x_offset, length) in enumerate(zip(spread, finger_lengths)):
        finger_extension = extension
        if shape_name == "TEST_POINT_UP" and f != 1:
            # everything except the index finger stays curled
            finger_extension = 0.25

        for j in range(length):
            t = (j + 1) / length
            points[idx] = [
                0.5 + base_x_offset * t,
                0.9 - t * 0.6 * finger_extension,
                0.0,
            ]
            idx += 1

    return points


def generate_synthetic_dataset(
    classes=TEST_CLASSES,
    samples_per_class: int = SAMPLES_PER_CLASS,
    noise_std: float = 0.02,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = ["label", "synthetic"] + [
        f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
    ]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_name in classes:
            base = _base_hand_shape(class_name, rng)
            for _ in range(samples_per_class):
                noisy = base + rng.normal(0, noise_std, size=base.shape)
                flat = noisy.reshape(-1)
                assert flat.shape[0] == FEATURE_LENGTH
                writer.writerow([class_name, True, *flat.tolist()])

    return OUTPUT_PATH


if __name__ == "__main__":
    path = generate_synthetic_dataset()
    print(f"Wrote SYNTHETIC TEST DATA (not ISL data) to: {path}")
    print(f"Classes: {TEST_CLASSES}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
