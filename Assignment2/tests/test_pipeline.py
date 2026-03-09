"""TDD tests for the end-to-end pipeline.

RED phase — defines the integration contract for FaceRecognitionPipeline.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPipelineInterface:
    """Verify the pipeline can be instantiated and returns correct types."""

    def test_import(self):
        from facerec.pipeline import FaceRecognitionPipeline, RecognitionResult
        assert FaceRecognitionPipeline is not None
        assert RecognitionResult is not None

    def test_instantiation(self):
        from facerec.pipeline import FaceRecognitionPipeline
        from facerec.config import Config
        cfg = Config.for_testing()
        pipe = FaceRecognitionPipeline(cfg)
        assert pipe is not None

    def test_process_image_returns_list(self, sample_face_image):
        from facerec.pipeline import FaceRecognitionPipeline
        from facerec.config import Config
        cfg = Config.for_testing()
        pipe = FaceRecognitionPipeline(cfg)
        results = pipe.process_image(sample_face_image)
        assert isinstance(results, list)

    def test_result_dataclass_fields(self):
        """RecognitionResult should have bbox, name, confidence, and embedding."""
        from facerec.pipeline import RecognitionResult
        r = RecognitionResult(
            bbox=np.array([10, 20, 100, 150]),
            name="test",
            confidence=0.85,
            embedding=np.zeros(512, dtype=np.float32),
        )
        assert r.name == "test"
        assert r.confidence == 0.85
        assert r.bbox.shape == (4,)
        assert r.embedding.shape == (512,)
