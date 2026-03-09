"""TDD tests for the face recognition / embedding module.

RED phase — defines the expected interface for FaceRecognizer.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestFaceRecognizerInterface:
    """Verify FaceRecognizer can be created and has the right API."""

    def test_import(self):
        """FaceRecognizer is importable."""
        from facerec.recognizer import FaceRecognizer
        assert FaceRecognizer is not None

    def test_instantiation(self):
        """Can create a FaceRecognizer with defaults."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        assert rec is not None

    def test_get_embedding_returns_512d(self, aligned_face_112):
        """Embedding from a 112×112 face should be a 512-d vector."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        emb = rec.get_embedding(aligned_face_112)
        assert emb.shape == (512,)

    def test_embedding_is_unit_normalised(self, aligned_face_112):
        """Returned embedding should be L2-normalised (norm ≈ 1.0)."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        emb = rec.get_embedding(aligned_face_112)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Embedding norm is {norm}, expected ~1.0"

    def test_embedding_dtype_is_float32(self, aligned_face_112):
        """Embedding should be float32 for efficient storage and comparison."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        emb = rec.get_embedding(aligned_face_112)
        assert emb.dtype == np.float32


class TestSimilarityComputation:
    """Verify cosine similarity helper."""

    def test_compute_similarity_same_vector(self, random_embedding):
        """Cosine similarity of a vector with itself should be ~1.0."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        sim = rec.compute_similarity(random_embedding, random_embedding)
        assert abs(sim - 1.0) < 1e-5

    def test_compute_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity ~0.0."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = rec.compute_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_compute_similarity_range(self, random_embedding):
        """Similarity should always be in [-1, 1]."""
        from facerec.recognizer import FaceRecognizer
        rec = FaceRecognizer()
        other = np.random.randn(512).astype(np.float32)
        other /= np.linalg.norm(other)
        sim = rec.compute_similarity(random_embedding, other)
        assert -1.0 <= sim <= 1.0
