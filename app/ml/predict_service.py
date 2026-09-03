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
"""

from __future__ import annotations

from typing import Optional

from app.ml.feature_extraction import extract_features
from app.ml.model import GestureMLP, model_files_exist


class PredictionUnavailableError(Exception):
    """Raised when there is no trained model to predict with."""


class GesturePredictionService:
    def __init__(self):
        self._model: Optional[GestureMLP] = None
        self._load_attempted = False

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
        self._ensure_loaded()

    def is_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except PredictionUnavailableError:
            return False

    def predict(self, landmarks) -> dict:
        """
        landmarks: the JSON-decoded list of 21 {x, y, z} dicts sent by
        the browser (MediaPipe JS output).

        Returns a dict describing the outcome. Always includes
        `status`, which is one of:
          - "ok"                 : a real prediction was made
          - "model_unavailable"  : no trained model exists yet
          - "invalid_input"      : landmarks were missing/malformed
        """
        try:
            self._ensure_loaded()
        except PredictionUnavailableError as exc:
            return {"status": "model_unavailable", "message": str(exc)}

        if not landmarks:
            return {
                "status": "invalid_input",
                "message": "No hand landmarks were provided.",
            }

        try:
            features = extract_features(landmarks)
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "status": "invalid_input",
                "message": f"Could not process landmarks: {exc}",
            }

        label, confidence = self._model.predict_single(features)
        is_synthetic = bool(
            self._model.metadata.get("is_synthetic_test_data", False)
        )

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


# Module-level singleton reused across requests.
prediction_service = GesturePredictionService()
