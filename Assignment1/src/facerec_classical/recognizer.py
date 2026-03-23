"""PCA + LDA face recognizer with SED matching and metrics.

Implements the Eigenfaces (PCA) → Fisherfaces (LDA) pipeline with
Squared Euclidean Distance to class centroids, plus ambiguity-based
open-set rejection for unknown faces.
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
        **kwargs: Any,
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

        # Per-class centroids in LDA space for ambiguity rejection
        self._class_centroids: dict[int, np.ndarray] = {}
        self._class_max_dist: dict[int, float] = {}

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

        # Compute per-class centroids and max within-class distances
        self._class_centroids = {}
        self._class_max_dist = {}
        for cls in np.unique(y_encoded):
            mask = y_encoded == cls
            class_vecs = X_lda[mask]
            centroid = np.mean(class_vecs, axis=0)
            self._class_centroids[int(cls)] = centroid
            # Max Euclidean distance of any training sample to its centroid
            dists = np.linalg.norm(class_vecs - centroid, axis=1)
            self._class_max_dist[int(cls)] = float(np.max(dists))

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
        """Project the face vector into the LDA subspace.

        1. Applies standard scalable `PCA.transform()`
        2. Applies Fisher `LinearDiscriminantAnalysis.transform()`

        Returns
        -------
        np.ndarray
            Flattened 1D array of extracted features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        pca_vec = self._pca.transform([face_vector.flatten()])
        lda_vec = self._lda.transform(pca_vec)
        return lda_vec.flatten()

    def predict(self, face_vector: np.ndarray) -> tuple[str, float]:
        """Predict identity using Sklearn One-Class SVM envelope and Cosine angle.

        Parameters
        ----------
        face_vector : np.ndarray
            Flattened face image, shape ``(n_features,)``.

        Returns
        -------
        tuple[str, float]
            ``(predicted_name, min_cos_dist)`` — ``"Unknown"`` if limits broken.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        test_proj = self.project(face_vector)

        # 1. Compute Euclidean distance to each class centroid
        centroid_dists: list[tuple[int, float]] = []
        for cls, centroid in self._class_centroids.items():
            dist = float(np.linalg.norm(test_proj - centroid))
            centroid_dists.append((cls, dist))

        # Sort by distance (closest first)
        centroid_dists.sort(key=lambda x: x[1])
        best_cls, best_dist = centroid_dists[0]
        second_cls, second_dist = centroid_dists[1] if len(centroid_dists) > 1 else (best_cls, best_dist)

        best_name = self._label_encoder.inverse_transform([best_cls])[0]  # type: ignore[union-attr]

        # 2. Ambiguity check: is the face clearly closer to one class?
        #    ratio > 1.5 means best class is meaningfully closer than second
        ambiguity_ratio = second_dist / best_dist if best_dist > 1e-9 else 1.0

        # 3. Within-class check: is the face within the known radius of the class?
        max_radius = self._class_max_dist.get(best_cls, 0.0)
        radius_ratio = best_dist / max_radius if max_radius > 1e-9 else float('inf')

        # Reject if: face is outside the known class radius OR no clear winner
        if radius_ratio > 1.1 or ambiguity_ratio < 1.5:
            reason = "radius" if radius_ratio > 1.1 else "ambiguity"
            print(f"  [UNKNOWN] nearest={best_name} r={radius_ratio:.2f} a={ambiguity_ratio:.2f} ({reason})")
            return ("Unknown", best_dist)

        print(f"  [MATCH ✓] {best_name} r={radius_ratio:.2f} a={ambiguity_ratio:.2f}")
        return (str(best_name), best_dist)

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
