"""
model.py

Part of the GestureSpeak AI ML pipeline.

A small, dependency-free (numpy-only) feed-forward neural network
("multi-layer perceptron") appropriate for classifying the 63-dim
normalized landmark feature vectors produced by feature_extraction.py.

We deliberately avoid adding scikit-learn / PyTorch / TensorFlow as a
new dependency (requirement: don't introduce unnecessary frameworks,
don't modify requirements.txt unless absolutely necessary). numpy is
already a project dependency (pulled in via mediapipe/opencv), so this
module implements forward pass, backpropagation, and gradient descent
directly with numpy.

Architecture:
    Input (63) -> Dense(hidden_dim) -> ReLU -> Dense(num_classes) -> Softmax

This is intentionally simple: with a small number of ISL sign classes
and a low-dimensional geometric feature vector, a single hidden layer
MLP is sufficient and fast enough to run in real time in a Flask
request/response cycle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.ml.feature_extraction import FEATURE_LENGTH

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "gesture_model.npz"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "gesture_model_meta.json"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class GestureMLP:
    """
    Single-hidden-layer MLP classifier over normalized landmark
    feature vectors.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_LENGTH,
        hidden_dim: int = 64,
        labels: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.labels: List[str] = labels or []

        rng = np.random.default_rng(seed)
        num_classes = max(len(self.labels), 1)

        # He initialization for ReLU hidden layer.
        self.W1 = rng.normal(
            0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)
        )
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(
            0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes)
        )
        self.b2 = np.zeros(num_classes)

        # metadata set by whoever trains/saves the model
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Forward pass / inference
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        probs = _softmax(z2)
        cache = (X, z1, a1, z2, probs)
        return probs, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Returns (predicted_label_list, confidence_array) for each row
        of X.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(indices)), indices]
        labels = [self.labels[i] for i in indices]
        return labels, confidences

    def predict_single(self, x: np.ndarray) -> Tuple[str, float]:
        labels, confidences = self.predict(x.reshape(1, -1))
        return labels[0], float(confidences[0])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y_indices: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2_reg: float = 1e-4,
        batch_size: int = 32,
        verbose: bool = True,
        seed: int = 0,
    ) -> List[float]:
        """
        Train the MLP with mini-batch gradient descent and a
        cross-entropy loss.

        X: (N, input_dim) feature matrix
        y_indices: (N,) integer class indices into self.labels

        Returns the per-epoch training loss history.
        """
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        n_classes = len(self.labels)

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_indices] = 1.0

        loss_history: List[float] = []

        for epoch in range(epochs):
            perm = rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_onehot[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                m = X_batch.shape[0]
                if m == 0:
                    continue

                probs, (Xc, z1, a1, z2, _) = self._forward(X_batch)

                # cross-entropy loss (+ L2 regularization)
                eps = 1e-9
                ce = -np.sum(y_batch * np.log(probs + eps)) / m
                l2 = l2_reg * (
                    np.sum(self.W1**2) + np.sum(self.W2**2)
                ) / 2
                loss = ce + l2
                epoch_loss += loss
                n_batches += 1

                # backpropagation
                dz2 = (probs - y_batch) / m  # (m, C)
                dW2 = a1.T @ dz2 + l2_reg * self.W2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)  # ReLU derivative
                dW1 = Xc.T @ dz1 + l2_reg * self.W1
                db1 = np.sum(dz1, axis=0)

                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                preds = np.argmax(self._forward(X)[0], axis=1)
                acc = float(np.mean(preds == y_indices))
                print(
                    f"epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  "
                    f"train_acc={acc:.3f}"
                )

        return loss_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            model_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )

        meta = dict(self.metadata)
        meta.update(
            {
                "labels": self.labels,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> "GestureMLP":
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python -m app.ml.train` first."
            )

        with open(metadata_path) as f:
            meta = json.load(f)

        instance = cls(
            input_dim=meta["input_dim"],
            hidden_dim=meta["hidden_dim"],
            labels=meta["labels"],
        )
        weights = np.load(model_path)
        instance.W1 = weights["W1"]
        instance.b1 = weights["b1"]
        instance.W2 = weights["W2"]
        instance.b2 = weights["b2"]
        instance.metadata = meta
        return instance


def model_files_exist(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> bool:
    return Path(model_path).exists() and Path(metadata_path).exists()
