"""
feature_extraction.py

Part of the GestureSpeak AI ML pipeline.

Converts the 21 raw (x, y, z) hand landmarks produced by MediaPipe
(either the browser JS "Hands" solution or the Python solution used
by app/services/hand_detector.py) into a fixed-length, normalized
feature vector suitable for a machine-learning classifier.

This module contains NO gesture-specific rules. It only performs
generic geometric normalization (translation + scale invariance) so
that the same hand shape produces (approximately) the same feature
vector regardless of where the hand is in the frame or how close it
is to the camera.

Feature vector layout:
    21 landmarks * 3 coordinates (x, y, z) = 63 features, ordered by
    MediaPipe landmark index 0..20, each landmark contributing
    (x, y, z) in that order.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z
FEATURE_LENGTH = NUM_LANDMARKS * NUM_COORDS  # 63

# MediaPipe landmark indices used as normalization anchors.
WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9


def _landmarks_to_array(landmarks) -> np.ndarray:
    """
    Normalize the many possible shapes `landmarks` can arrive in into
    a (21, 3) numpy array.

    Accepts:
      - a list/tuple of 21 dicts: [{"x":.., "y":.., "z":..}, ...]
      - a list/tuple of 21 objects with .x/.y/.z attributes (e.g. the
        MediaPipe Python `NormalizedLandmark` objects)
      - a list/tuple of 21 (x, y, z) sequences
      - an already-built (21, 3) or (63,) numpy array / list
    """
    if isinstance(landmarks, np.ndarray):
        arr = landmarks
    elif len(landmarks) == FEATURE_LENGTH and all(
        isinstance(v, (int, float)) for v in landmarks
    ):
        arr = np.array(landmarks, dtype=np.float64)
    else:
        rows = []
        for point in landmarks:
            if isinstance(point, dict):
                rows.append([point["x"], point["y"], point.get("z", 0.0)])
            elif hasattr(point, "x") and hasattr(point, "y"):
                rows.append([point.x, point.y, getattr(point, "z", 0.0)])
            else:
                # assume it's an (x, y, z) or (x, y) sequence
                point = list(point)
                if len(point) == 2:
                    point.append(0.0)
                rows.append(point[:3])
        arr = np.array(rows, dtype=np.float64)

    arr = arr.reshape(-1)
    if arr.size != FEATURE_LENGTH:
        raise ValueError(
            f"Expected {NUM_LANDMARKS} landmarks with {NUM_COORDS} "
            f"coordinates each ({FEATURE_LENGTH} values total), got "
            f"{arr.size} values."
        )
    return arr.reshape(NUM_LANDMARKS, NUM_COORDS)


def extract_features(landmarks) -> np.ndarray:
    """
    Convert raw hand landmarks into a normalized 63-dim feature vector.

    Normalization steps:
      1. Translation invariance: subtract the wrist landmark (index 0)
         from every point, so the wrist becomes the origin (0, 0, 0).
      2. Scale invariance: divide every coordinate by the Euclidean
         distance between the wrist and the middle-finger MCP joint
         (landmark 9), a stable "palm size" reference. This makes the
         feature vector roughly invariant to hand distance from the
         camera.

    Returns:
        np.ndarray of shape (63,), dtype float64.
    """
    points = _landmarks_to_array(landmarks)

    wrist = points[WRIST_INDEX].copy()
    centered = points - wrist

    palm_reference = centered[MIDDLE_FINGER_MCP_INDEX]
    scale = float(np.linalg.norm(palm_reference))
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero on degenerate input

    normalized = centered / scale

    return normalized.reshape(-1).astype(np.float64)


def extract_features_batch(landmarks_list) -> np.ndarray:
    """
    Apply extract_features() to a list of landmark sets.

    Returns:
        np.ndarray of shape (N, 63).
    """
    return np.stack([extract_features(lm) for lm in landmarks_list])
