"""
sequence_utils.py

Part of the GestureSpeak AI ML pipeline.

STAGE 3 CORRECTION (see diagnostic report):
--------------------------------------------
The original Stage 3 design converted every example -- static or
dynamic -- into a fixed 20-frame window (repeating a static frame 20x,
or resampling a dynamic sequence to 20 frames) and flattened it to
1260 features. Diagnosed and CONFIRMED root cause of that design's
64.4% synthetic accuracy (vs. 100% at 63 dims): repeating one frame's
feature vector 20 IDENTICAL times means backprop computes the exact
same gradient for all 20 corresponding weight blocks in the first
Dense layer every step, which is mathematically equivalent to a ~20x
higher effective learning rate on that layer. This caused the
optimizer to overshoot/oscillate instead of converge (confirmed by
reproducing 100% train accuracy at lr/20, and by showing MORE hidden
capacity at the original lr made results WORSE, not better -- the
signature of an optimization instability, not a capacity or data
problem).

NEW DESIGN: a compact, fixed-length (189-dim) representation built
from three per-frame feature vectors (each individually 63-dim,
normalized by feature_extraction.extract_features -- unchanged):

    pose    = features of the FIRST captured frame
              (the hand shape -- this is the whole signal for a
              static sign)
    delta   = features of the LAST frame  minus the FIRST frame
              (net displacement over the example -- ~0 for a static
              sign held still, non-zero and direction-carrying for a
              dynamic sign)
    summary = (mean of all frames' features) minus the FIRST frame
              (a cheap summary of the overall path shape/curvature,
              not just the endpoints -- also ~0 for static)

    compact_features = concatenate(pose, delta, summary)  -> 189 dims

This lets STATIC and DYNAMIC examples coexist in one model without
any special-casing: a static example (T=1 frame) naturally produces
delta=0 and summary=0 vectors (not IDENTICAL data, i.e. no duplicate
columns to trigger the gradient-duplication pathology -- they're
exact zeros, which simply contribute zero gradient for those
examples, which is correct and harmless). A dynamic example's delta
and summary carry the actual real motion signal. No resampling to a
fixed number of frames is needed at all -- pose/delta/summary are
well-defined for any sequence length T >= 1, so this also removes the
interpolation step entirely for the default training path.

This module contains NO gesture-specific rules -- only generic,
label-agnostic feature construction.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from app.ml.feature_extraction import FEATURE_LENGTH, extract_features

# pose (63) + delta (63) + summary (63) = 189
COMPACT_FEATURE_DIM = 3 * FEATURE_LENGTH


def build_compact_features(
    raw_frames: Sequence,
    example_type: str,
) -> np.ndarray:
    """
    Convert one example's raw landmark frame(s) into a single
    189-dim compact feature vector ready for GestureMLP.

    raw_frames: a non-empty, time-ordered sequence of raw landmark
        sets, each acceptable by feature_extraction.extract_features()
        (e.g. a list of 63 floats, a list of 21 {x,y,z} dicts, etc.).
        For a static example this is a single-item sequence.
    example_type: "static" or "dynamic" (case-insensitive). Used only
        as a sanity check against the frame count -- collect_data.py
        guarantees a static example is exactly 1 frame.

    Returns:
        np.ndarray of shape (COMPACT_FEATURE_DIM,) = (189,).
    """
    if not raw_frames:
        raise ValueError("build_compact_features() requires at least one frame.")

    example_type = example_type.strip().lower()
    if example_type not in ("static", "dynamic"):
        raise ValueError(
            f"Unknown example_type {example_type!r}; expected 'static' or 'dynamic'."
        )
    if example_type == "static" and len(raw_frames) != 1:
        raise ValueError(
            f"A 'static' example must have exactly 1 frame, got {len(raw_frames)}."
        )

    per_frame_features = np.stack([extract_features(frame) for frame in raw_frames])

    pose = per_frame_features[0]
    delta = per_frame_features[-1] - per_frame_features[0]
    summary = per_frame_features.mean(axis=0) - per_frame_features[0]

    return np.concatenate([pose, delta, summary])


def group_rows_by_example(rows: List[dict]) -> "dict[str, List[dict]]":
    """
    Group CSV DictReader rows (from the Stage 2 collect_data.py
    format) by `example_id`, preserving each group's original row
    order (callers should still sort by frame_index for safety, which
    load_dataset() in train.py does).
    """
    groups: "dict[str, List[dict]]" = {}
    for row in rows:
        groups.setdefault(row["example_id"], []).append(row)
    return groups