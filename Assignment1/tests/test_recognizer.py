"""TDD tests for the PCA+LDA recognizer."""

from __future__ import annotations

import numpy as np
import pytest


def _make_training_data(n_people: int = 5, n_per_person: int = 8, dim: int = 100):
    """Generate synthetic training data: clusters of face vectors."""
    rng = np.random.RandomState(42)
    X = []
    y = []
    for person in range(n_people):
        centre = rng.randn(dim) * 10 + person * 20
        for _ in range(n_per_person):
            X.append(centre + rng.randn(dim) * 2)
            y.append(f"Person_{person:02d}")
    return np.array(X), np.array(y)


class TestPCALDARecognizerInterface:
    """Verify PCALDARecognizer API."""

    def test_import(self):
        from facerec_classical.recognizer import PCALDARecognizer
        assert PCALDARecognizer is not None

    def test_instantiation(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer()
        assert rec is not None
        assert not rec.is_fitted


class TestFit:
    """Tests for fit() method."""

    def test_fit_returns_metrics_dict(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        metrics = rec.fit(X, y)

        assert isinstance(metrics, dict)
        assert "n_pca_components" in metrics
        assert "n_lda_components" in metrics
        assert "pca_explained_var" in metrics
        assert "reconstruction_error" in metrics
        assert "train_accuracy" in metrics
        assert "n_samples" in metrics
        assert "n_classes" in metrics

    def test_fit_sets_fitted_flag(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        rec.fit(X, y)
        assert rec.is_fitted

    def test_explained_variance_is_valid(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        metrics = rec.fit(X, y)

        var_ratios = metrics["pca_explained_var"]
        assert len(var_ratios) > 0
        assert all(0 <= v <= 1 for v in var_ratios)
        assert metrics["cumulative_variance"] <= 1.0

    def test_reconstruction_error_is_nonnegative(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        metrics = rec.fit(X, y)
        assert metrics["reconstruction_error"] >= 0


class TestProject:
    """Tests for project() method."""

    def test_project_returns_reduced_vector(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        rec.fit(X, y)

        projected = rec.project(X[0])
        assert projected.ndim == 1
        # LDA dims = n_classes - 1 = 4
        assert len(projected) == 4

    def test_project_before_fit_raises(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            rec.project(np.zeros(100))


class TestPredict:
    """Tests for predict() method."""

    def test_predict_training_sample_returns_correct_label(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10, sed_threshold=1.0)
        X, y = _make_training_data()
        rec.fit(X, y)

        name, dist = rec.predict(X[0])
        assert name == y[0]
        assert dist >= 0

    def test_predict_returns_unknown_for_far_vector(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10, sed_threshold=0.001)
        X, y = _make_training_data()
        rec.fit(X, y)

        # Random vector far from any cluster
        rng = np.random.RandomState(99)
        far_vec = rng.randn(100) * 1000
        name, dist = rec.predict(far_vec)
        assert name == "Unknown"


class TestEvaluate:
    """Tests for evaluate() method."""

    def test_evaluate_returns_accuracy_dict(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10, sed_threshold=1.0)
        X, y = _make_training_data()
        rec.fit(X, y)

        result = rec.evaluate(X, y)
        assert "accuracy" in result
        assert "n_correct" in result
        assert "n_total" in result
        assert "per_class_accuracy" in result

    def test_evaluate_train_accuracy_is_high(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10, sed_threshold=1.0)
        X, y = _make_training_data()
        rec.fit(X, y)

        result = rec.evaluate(X, y)
        assert result["accuracy"] >= 0.8


class TestGetExplainedVariance:
    """Tests for get_explained_variance()."""

    def test_returns_array(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer(n_components_pca=10)
        X, y = _make_training_data()
        rec.fit(X, y)

        var = rec.get_explained_variance()
        assert isinstance(var, np.ndarray)
        assert len(var) > 0

    def test_before_fit_raises(self):
        from facerec_classical.recognizer import PCALDARecognizer
        rec = PCALDARecognizer()
        with pytest.raises(RuntimeError):
            rec.get_explained_variance()
