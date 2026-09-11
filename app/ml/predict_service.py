"""
predict_service.py

Part of the GestureSpeak AI ML pipeline.

Thin wrapper used by the Flask layer (app/routes/main.py) to turn raw
browser-side MediaPipe landmarks into a prediction.

Design goals:
  - Load the trained model ONCE (lazily, on first request) and reuse
    it, rather than reloading from disk on every /predict call.
  - If no trained model file exists yet, do NOT fake a prediction.
    Report clearly that a trained model is unavailable so the frontend
    can show an honest message instead of a made-up gesture name.
  - If the loaded model was trained on synthetic TEST DATA only, flag
    that in every response so nobody mistakes a technical pipeline
    test for real ISL recognition.

STAGE 4 SEGMENTATION UPGRADE
----------------------------
The previous Stage 4 implementation fed an UNSEGMENTED rolling
20-frame window straight into build_compact_features() on every call.
That meant "pose" was just whichever frame happened to be oldest in
the window at that instant -- not the true start of a gesture -- so a
dynamic sign could be represented by an arbitrary slice of continuous
motion rather than a complete gesture, and a static sign could never
be distinguished from "hand happens to be still for 20 frames."

This version replaces the raw rolling window with a small, generic,
label-independent STATE MACHINE (see GestureState below) that:
  - tracks whether a hand is present at all,
  - detects the START of a gesture from genuine motion (with
    hysteresis so single-frame jitter can't trigger it),
  - collects exactly the frames belonging to that gesture (seeded with
    a short pre-motion lookback so the true onset isn't lost to
    threshold-crossing delay),
  - detects the END of a gesture from renewed stability (again with
    hysteresis),
  - ALSO recognizes a held-still pose as a valid STATIC gesture
    candidate after a configurable hold period, with a cooldown so it
    doesn't spam predictions from an unchanged hand,
  - and only THEN calls the existing, unmodified
    sequence_utils.build_compact_features() on the captured frame
    sequence -- there is no second feature-extraction algorithm here,
    only a decision about WHEN to call the existing one and WHICH
    frames to hand it.

None of this contains gesture-specific rules: motion is measured only
as the generic L2 distance between consecutive frames' existing
normalized feature_extraction.extract_features() vectors, compared
against a single tunable threshold that applies identically to every
sign.

RESPONSE CONTRACT
------------------
The four original fields are unchanged for a real prediction:
    status="ok", prediction, confidence, is_test_model, warning
"model_unavailable" and "invalid_input" behave exactly as before.

ONE new status is introduced: "tracking" -- returned whenever a valid
hand frame was processed but the state machine has not (yet) finalized
a gesture on this call (still building a stability baseline, still
collecting an in-progress motion, in a post-prediction cooldown, or
just discarded a noise/too-long/interrupted attempt). It carries a
human-readable `message` describing the sub-state, the same shape
previously used for "insufficient_frames" (which this status
supersedes and replaces -- there is no longer a fixed frame-count
gate; the same "tracking" status now also covers "actively collecting
a gesture" and "waiting out a cooldown").

LIMITATIONS (flagged, not solved here -- see the accompanying report):
this is still a single, per-process (not per-session) state machine,
and the initial threshold/frame-count parameters below are
placeholders to be tuned once real ISL data exists.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from app.ml.feature_extraction import extract_features
from app.ml.model import GestureMLP, model_files_exist
from app.ml.sequence_utils import build_compact_features

# ============================================================
# SEGMENTATION PARAMETERS
#
# These are INITIAL, GENERIC PLACEHOLDER VALUES chosen to be
# conservative and label-independent -- they are NOT tuned against
# real ISL data (there isn't any yet). Revisit every value here once
# the real 10-sign dataset has been collected and real motion/jitter
# magnitudes are known.
# ============================================================

# L2 distance, in normalized 63-dim extract_features() space, between
# two consecutive frames above which a frame counts as "moving" rather
# than "stable". Chosen to sit comfortably above ordinary MediaPipe
# landmark jitter on a held-still hand, based on the noise levels used
# in the existing synthetic test data (noise_std=0.02) -- NOT measured
# against a real camera/hand yet.
MOTION_THRESHOLD = 0.15

# Consecutive "moving" frames required to confirm a gesture has
# actually started (hysteresis against a single noisy frame).
MOTION_START_CONSEC_FRAMES = 3

# Consecutive "stable" frames required, once a gesture is in progress,
# to confirm it has ended.
MOTION_END_CONSEC_FRAMES = 4

# Consecutive "stable" frames required, while NOT in an active
# gesture, before a held pose becomes a valid STATIC prediction
# candidate.
STATIC_HOLD_FRAMES = 8

# After emitting a prediction (static or dynamic), how many subsequent
# stable frames must pass before another STATIC prediction can be
# emitted from the same held pose. Prevents spamming predictions for
# an unchanged hand.
STATIC_PREDICTION_COOLDOWN_FRAMES = 15

# A completed motion episode shorter than this many frames is treated
# as noise (a twitch, not a gesture) and discarded rather than
# predicted on.
MIN_GESTURE_FRAMES = 3

# Hard cap on how many frames an in-progress gesture may accumulate
# before it is discarded outright (see _handle_active_frame). This is
# a safety limit, not a tuned "typical gesture length".
MAX_GESTURE_FRAMES = 40

# How many recent frames are always kept so that, when a gesture start
# is confirmed, the capture can be seeded with the frames immediately
# BEFORE the threshold was crossed -- otherwise the first
# MOTION_START_CONSEC_FRAMES frames of real motion would be the only
# "beginning" captured, losing context right at the true onset.
PRE_MOTION_LOOKBACK_FRAMES = 5

# Consecutive missing/invalid-hand frames required before the state
# machine resets (clearing any in-progress gesture) rather than
# tolerating a brief tracking dropout.
HAND_MISSING_FRAMES_TO_RESET = 5


class GestureState:
    """String constants for the segmentation state machine."""
    IDLE = "idle"       # no hand seen yet / just reset
    STABLE = "stable"   # hand present, not currently mid-gesture
    ACTIVE = "active"   # a motion episode is being captured


class PredictionUnavailableError(Exception):
    """Raised when there is no trained model to predict with."""


class GesturePredictionService:
    def __init__(self):
        self._model: Optional[GestureMLP] = None
        self._load_attempted = False

        # --- segmentation state (see GestureState) ---
        self._state = GestureState.IDLE
        self._last_features: Optional[np.ndarray] = None
        self._lookback_buffer = deque(maxlen=PRE_MOTION_LOOKBACK_FRAMES)
        self._active_frames: list = []
        self._stable_streak = 0
        self._moving_streak = 0
        self._missing_streak = 0
        self._cooldown_remaining = 0

    # ------------------------------------------------------------------
    # Model loading (unchanged behavior)
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not model_files_exist():
            raise PredictionUnavailableError(
                "No trained model is available yet. Train one with: "
                "python -m app.ml.generate_test_data && "
                "python -m app.ml.train --data app/ml/data/synthetic_test_data.csv"
            )
        self._model = GestureMLP.load()

    def reload(self):
        """Force a reload from disk (e.g. after re-training)."""
        self._model = None
        self._reset_state()
        self._ensure_loaded()

    def reset_buffer(self):
        """Backward-compatible alias for _reset_state()."""
        self._reset_state()

    def is_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except PredictionUnavailableError:
            return False

    # ------------------------------------------------------------------
    # Segmentation state machine
    # ------------------------------------------------------------------
    def _reset_state(self):
        self._state = GestureState.IDLE
        self._last_features = None
        self._lookback_buffer.clear()
        self._active_frames = []
        self._stable_streak = 0
        self._moving_streak = 0
        self._missing_streak = 0
        self._cooldown_remaining = 0

    def _handle_missing_hand(self, message: str) -> dict:
        self._missing_streak += 1
        if self._missing_streak >= HAND_MISSING_FRAMES_TO_RESET:
            discarded_active = (
                self._state == GestureState.ACTIVE and len(self._active_frames) > 0
            )
            self._reset_state()
            if discarded_active:
                message = message + " In-progress gesture discarded because the hand was lost."
        return {"status": "invalid_input", "message": message}

    def _process_valid_frame(self, landmarks, features: np.ndarray) -> dict:
        self._missing_streak = 0
        motion = (
            0.0
            if self._last_features is None
            else float(np.linalg.norm(features - self._last_features))
        )
        self._lookback_buffer.append(landmarks)

        if self._state == GestureState.IDLE:
            self._state = GestureState.STABLE
            self._stable_streak = 1
            self._moving_streak = 0
            result = {"status": "tracking", "message": "Hand detected; establishing baseline."}
        elif self._state == GestureState.STABLE:
            result = self._handle_stable_frame(landmarks, motion)
        else:  # GestureState.ACTIVE
            result = self._handle_active_frame(landmarks, motion)

        self._last_features = features
        return result

    def _handle_stable_frame(self, landmarks, motion: float) -> dict:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if motion >= MOTION_THRESHOLD:
            self._moving_streak += 1
            self._stable_streak = 0
        else:
            self._moving_streak = 0
            self._stable_streak += 1

        if self._moving_streak >= MOTION_START_CONSEC_FRAMES:
            # Gesture start confirmed. Seed the capture with the
            # pre-motion lookback so the true onset (the frames right
            # before the threshold was crossed) isn't lost.
            self._active_frames = list(self._lookback_buffer)
            self._state = GestureState.ACTIVE
            self._moving_streak = 0
            self._stable_streak = 0
            return {"status": "tracking", "message": "Motion detected; collecting gesture."}

        if self._stable_streak >= STATIC_HOLD_FRAMES and self._cooldown_remaining == 0:
            result = self._finalize_prediction([landmarks], "static")
            self._cooldown_remaining = STATIC_PREDICTION_COOLDOWN_FRAMES
            self._stable_streak = 0
            return result

        return {"status": "tracking", "message": "Hand stable; monitoring."}

    def _handle_active_frame(self, landmarks, motion: float) -> dict:
        self._active_frames.append(landmarks)

        if motion < MOTION_THRESHOLD:
            self._stable_streak += 1
        else:
            self._stable_streak = 0

        if self._stable_streak >= MOTION_END_CONSEC_FRAMES:
            captured = list(self._active_frames)
            self._state = GestureState.STABLE
            self._active_frames = []
            self._stable_streak = 0
            self._moving_streak = 0

            if len(captured) < MIN_GESTURE_FRAMES:
                return {
                    "status": "tracking",
                    "message": (
                        f"Motion too brief ({len(captured)} frame(s)); "
                        "discarded as noise, not a gesture."
                    ),
                }

            result = self._finalize_prediction(captured, "dynamic")
            self._cooldown_remaining = STATIC_PREDICTION_COOLDOWN_FRAMES
            return result

        if len(self._active_frames) >= MAX_GESTURE_FRAMES:
            # Conservative choice (documented, not "optimal"): discard
            # rather than predict on a capture that never showed a
            # clear end. Predicting on an arbitrarily-truncated
            # gesture risks a confident-looking but meaningless label;
            # discarding just costs the user one retry.
            self._state = GestureState.STABLE
            self._active_frames = []
            self._stable_streak = 0
            self._moving_streak = 0
            return {
                "status": "tracking",
                "message": (
                    f"Gesture exceeded {MAX_GESTURE_FRAMES} frames without a "
                    "detected end; discarded."
                ),
            }

        return {
            "status": "tracking",
            "message": f"Collecting gesture ({len(self._active_frames)} frame(s) so far).",
        }

    def _finalize_prediction(self, raw_frames: list, example_type: str) -> dict:
        """
        Build the SAME 189-dim compact feature vector training uses,
        via the existing, unmodified build_compact_features(), and run
        the existing, unmodified model.predict_single() on it.
        """
        features = build_compact_features(raw_frames, example_type)
        label, confidence = self._model.predict_single(features)
        is_synthetic = bool(self._model.metadata.get("is_synthetic_test_data", False))
        return {
            "status": "ok",
            "prediction": label,
            "confidence": confidence,
            "is_test_model": is_synthetic,
            "warning": (
                "This model was trained on SYNTHETIC TEST DATA, not "
                "real ISL signs. Predictions are for pipeline testing "
                "only."
                if is_synthetic
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def predict(self, landmarks) -> dict:
        """
        landmarks: the JSON-decoded list of 21 {x, y, z} dicts sent by
        the browser (MediaPipe JS output) for the CURRENT frame only.
        Each call advances the internal gesture-segmentation state
        machine by exactly one frame.

        Returns a dict describing the outcome. Always includes
        `status`, which is one of:
          - "ok"              : a real prediction was made (static or
                                 dynamic gesture completed)
          - "model_unavailable": no trained model exists yet
          - "invalid_input"    : landmarks were missing/malformed (also
                                 used to detect "no hand present")
          - "tracking"         : a valid hand frame was processed but
                                 no gesture was finalized this call
                                 (still baselining, still collecting an
                                 in-progress gesture, in a
                                 post-prediction cooldown, or an
                                 attempt was just discarded as noise /
                                 too long / interrupted)
        """
        try:
            self._ensure_loaded()
        except PredictionUnavailableError as exc:
            return {"status": "model_unavailable", "message": str(exc)}

        if not landmarks:
            return self._handle_missing_hand("No hand landmarks were provided.")

        try:
            features = extract_features(landmarks)
        except (ValueError, KeyError, TypeError) as exc:
            return self._handle_missing_hand(f"Could not process landmarks: {exc}")

        return self._process_valid_frame(landmarks, features)


# Module-level singleton reused across requests.
prediction_service = GesturePredictionService()