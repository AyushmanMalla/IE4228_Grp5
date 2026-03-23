"""PCA + LDA face recognizer with two-stage triage rejection.

Implements the Eigenfaces (PCA) → Fisherfaces (LDA) pipeline with
a cascaded rejection strategy:
  Stage 1: PCA reconstruction error filters non-gallery faces.
  Stage 2: Multi-class SVM probability in LDA space for identity
           resolution and final unknown rejection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class PCALDARecognizer:
    """Classical face recognizer using PCA dimensionality reduction
    followed by LDA for class separation, with two-stage triage rejection.

    Parameters
    ----------
    n_components_pca : int
        Number of PCA components to retain.
    n_components_lda : str | int
        Number of LDA components. ``"auto"`` = ``min(n_classes - 1, n_pca_dims)``.
    reconstruction_threshold : float
        Maximum PCA reconstruction MSE for Stage 1 acceptance.
    svm_prob_threshold : float
        Minimum multi-class SVM probability for Stage 2 acceptance.
    sed_threshold : float
        Legacy threshold, kept for backwards compatibility.
    """

    def __init__(
        self,
        n_components_pca: int = 50,
        n_components_lda: str | int = "auto",
        reconstruction_threshold: float = 5000.0,
        svm_prob_threshold: float = 0.60,
        sed_threshold: float = 0.45,
    ) -> None:
        self._n_pca = n_components_pca
        self._n_lda = n_components_lda
        self._recon_threshold = reconstruction_threshold
        self._svm_prob_threshold = svm_prob_threshold
        self._sed_threshold = sed_threshold  # legacy

        self._pca: PCA | None = None
        self._lda: LDA | None = None
        self._label_encoder: LabelEncoder | None = None
        self._svm: SVC | None = None

        # Stored training projections for NN matching fallback
        self._train_projected: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return self._pca is not None and self._lda is not None and self._svm is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train PCA → LDA pipeline with two-stage triage.

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
        reconstruction_error = float(
            np.mean((X.astype(np.float64) - X_reconstructed) ** 2)
        )

        # --- LDA ---
        if self._n_lda == "auto":
            n_lda = min(n_classes - 1, X_pca.shape[1])
        else:
            n_lda = int(self._n_lda)
        if n_lda < 1:
            n_lda = 1

        self._lda = LDA(n_components=n_lda)
        X_lda = self._lda.fit_transform(X_pca, y_encoded)

        # --- Two-Stage Triage: Multi-class SVM with probability ---
        self._svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        self._svm.fit(X_lda, y_encoded)

        # Store training projections
        self._train_projected = X_lda
        self._train_labels = y_encoded

        # Training accuracy (NN on training set)
        train_accuracy = self._compute_nn_accuracy(X_lda, y_encoded)

        return {
            "n_pca_components": n_pca,
            "n_lda_components": n_lda,
            "pca_explained_var": self._pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(
                np.sum(self._pca.explained_variance_ratio_)
            ),
            "reconstruction_error": reconstruction_error,
            "train_accuracy": train_accuracy,
            "n_samples": n_samples,
            "n_classes": n_classes,
        }

    def reconstruction_error(self, face_vector: np.ndarray) -> float:
        """Compute PCA reconstruction MSE for a single face vector.

        Parameters
        ----------
        face_vector : np.ndarray
            Flattened feature vector.

        Returns
        -------
        float
            Mean squared error between original and PCA-reconstructed vector.
        """
        if self._pca is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        vec = face_vector.flatten().astype(np.float64)
        pca_vec = self._pca.transform([vec])
        reconstructed = self._pca.inverse_transform(pca_vec).flatten()
        return float(np.mean((vec - reconstructed) ** 2))

    def project(self, face_vector: np.ndarray) -> np.ndarray:
        """Project the face vector into the LDA subspace.

        1. Applies PCA.transform()
        2. Applies LDA.transform()

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
        """Two-stage triage prediction.

        Stage 1: PCA reconstruction error → reject if above threshold.
        Stage 2: Multi-class SVM probability in LDA space → identity or reject.

        Parameters
        ----------
        face_vector : np.ndarray
            Flattened face feature vector, shape ``(n_features,)``.

        Returns
        -------
        tuple[str, float]
            ``(predicted_name, probability)`` — ``"Unknown"`` if rejected.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # --- Stage 1: PCA Reconstruction Error ---
        recon_err = self.reconstruction_error(face_vector)
        if recon_err > self._recon_threshold:
            return ("Unknown", 0.0)

        # --- Stage 2: Multi-class SVM Probability ---
        lda_vec = self.project(face_vector)
        
        # Get probability distributions
        probs = self._svm.predict_proba([lda_vec])[0]  # type: ignore[union-attr]
        max_prob = float(np.max(probs))
        
        if max_prob < self._svm_prob_threshold:
            return ("Unknown", max_prob)

        best_label = self._svm.classes_[np.argmax(probs)]  # type: ignore[union-attr]
        name = self._label_encoder.inverse_transform([best_label])[0]  # type: ignore[union-attr]
        return (str(name), max_prob)

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
