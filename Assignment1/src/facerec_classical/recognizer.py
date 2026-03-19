"""PCA + LDA face recognizer with SED matching and metrics.

Implements the Eigenfaces (PCA) → Fisherfaces (LDA) pipeline with
Squared Euclidean Distance nearest-neighbour classification and
training/testing metric reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder


class PCALDARecognizer:
    """Classical face recognizer using PCA dimensionality reduction
    followed by LDA for class separation.

    Parameters
    ----------
    n_components_pca : int
        Number of PCA components to retain.
    n_components_lda : str | int
        Number of LDA components. ``"auto"`` = ``min(n_classes - 1, n_pca_dims)``.
    sed_threshold : float
        Maximum SED for a positive match.
    """

    def __init__(
        self,
        n_components_pca: int = 50,
        n_components_lda: str | int = "auto",
        sed_threshold: float = 0.45,
    ) -> None:
        self._n_pca = n_components_pca
        self._n_lda = n_components_lda
        self._sed_threshold = sed_threshold

        self._pca: PCA | None = None
        self._lda: LDA | None = None
        self._label_encoder: LabelEncoder | None = None

        # Stored training projections for NN matching
        self._train_projected: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return self._pca is not None and self._lda is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train PCA → LDA pipeline on flattened face vectors.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Labels of shape ``(n_samples,)`` — string or int.

        Returns
        -------
        dict
            Training metrics including explained variance, reconstruction
            error, and component counts.
        """
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        n_classes = len(np.unique(y_encoded))
        n_samples = len(X)

        # --- PCA ---
        n_pca = min(self._n_pca, n_samples - n_classes)
        if n_pca < 1:
            n_pca = 1

        self._pca = PCA(n_components=n_pca, whiten=True)
        X_pca = self._pca.fit_transform(X)

        # Reconstruction error (MSE loss)
        X_reconstructed = self._pca.inverse_transform(X_pca)
        reconstruction_error = float(np.mean((X.astype(np.float64) - X_reconstructed) ** 2))

        # --- LDA ---
        if self._n_lda == "auto":
            n_lda = min(n_classes - 1, X_pca.shape[1])
        else:
            n_lda = int(self._n_lda)
        if n_lda < 1:
            n_lda = 1

        self._lda = LDA(n_components=n_lda)
        X_lda = self._lda.fit_transform(X_pca, y_encoded)

        # Store training projections
        self._train_projected = X_lda
        self._train_labels = y_encoded

        # Training accuracy (NN on training set)
        train_accuracy = self._compute_nn_accuracy(X_lda, y_encoded)

        return {
            "n_pca_components": n_pca,
            "n_lda_components": n_lda,
            "pca_explained_var": self._pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.sum(self._pca.explained_variance_ratio_)),
            "reconstruction_error": reconstruction_error,
            "train_accuracy": train_accuracy,
            "n_samples": n_samples,
            "n_classes": n_classes,
        }

    def project(self, face_vector: np.ndarray) -> np.ndarray:
        """Project a flattened face vector into PCA→LDA space.

        Parameters
        ----------
        face_vector : np.ndarray
            Shape ``(n_features,)`` or ``(1, n_features)``.

        Returns
        -------
        np.ndarray
            Projected vector in LDA space.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        vec = face_vector.reshape(1, -1)
        pca_vec = self._pca.transform(vec)  # type: ignore[union-attr]
        lda_vec = self._lda.transform(pca_vec)  # type: ignore[union-attr]
        return lda_vec.flatten()

    def predict(self, face_vector: np.ndarray) -> tuple[str, float]:
        """Predict identity using Cosine nearest-neighbour matching.

        Parameters
        ----------
        face_vector : np.ndarray
            Flattened face image, shape ``(n_features,)``.

        Returns
        -------
        tuple[str, float]
            ``(predicted_name, cosine_dist)`` — ``"Unknown"`` if dist > threshold.
        """
        from scipy.spatial.distance import cosine
        
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        test_proj = self.project(face_vector)

        min_dist = float("inf")
        best_idx = -1

        for i, train_vec in enumerate(self._train_projected):  # type: ignore[union-attr]
            # Handle empty or zero vectors
            if np.linalg.norm(test_proj) == 0 or np.linalg.norm(train_vec) == 0:
                continue
            
            dist = float(cosine(test_proj.flatten(), train_vec.flatten()))
            if dist < min_dist:
                min_dist = dist
                best_idx = self._train_labels[i]  # type: ignore[index]

        if min_dist < self._sed_threshold:
            name = self._label_encoder.inverse_transform([best_idx])[0]  # type: ignore[union-attr]
            return (str(name), min_dist)
        else:
            return ("Unknown", min_dist)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        """Evaluate on a test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test data, shape ``(n_samples, n_features)``.
        y_test : np.ndarray
            True labels.

        Returns
        -------
        dict
            ``{"accuracy", "n_correct", "n_total", "per_class_accuracy"}``.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_correct = 0
        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}

        for i in range(len(X_test)):
            pred_name, _ = self.predict(X_test[i])
            true_name = str(y_test[i])

            class_total[true_name] = class_total.get(true_name, 0) + 1
            if pred_name == true_name:
                n_correct += 1
                class_correct[true_name] = class_correct.get(true_name, 0) + 1

        n_total = len(X_test)
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        per_class = {
            name: class_correct.get(name, 0) / class_total[name]
            for name in class_total
        }

        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_class_accuracy": per_class,
        }

    def get_explained_variance(self) -> np.ndarray:
        """Return PCA explained variance ratios."""
        if self._pca is None:
            raise RuntimeError("Model not fitted.")
        return self._pca.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_nn_accuracy(projected: np.ndarray, labels: np.ndarray) -> float:
        """Leave-one-out NN accuracy on the training set."""
        n = len(projected)
        correct = 0
        for i in range(n):
            min_dist = float("inf")
            best_label = -1
            for j in range(n):
                if i == j:
                    continue
                dist = float(np.sum((projected[i] - projected[j]) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    best_label = labels[j]
            if best_label == labels[i]:
                correct += 1
        return correct / n if n > 0 else 0.0
