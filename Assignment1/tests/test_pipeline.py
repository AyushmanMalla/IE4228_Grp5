"""TDD tests for the end-to-end classical face recognition pipeline."""

from __future__ import annotations

import numpy as np
import pytest


class TestPipelineInterface:
    """Verify ClassicalFaceRecPipeline API."""

    def test_import(self):
        from facerec_classical.pipeline import ClassicalFaceRecPipeline
        assert ClassicalFaceRecPipeline is not None

    def test_result_dataclass(self):
        from facerec_classical.pipeline import RecognitionResult
        r = RecognitionResult(bbox=(10, 20, 50, 50), name="Alice", distance=1.5)
        assert r.name == "Alice"
        assert r.distance == 1.5


class TestPipelineTrain:
    """Tests for train() method."""

    def test_train_returns_metrics(self, tmp_dataset_with_faces):
        from facerec_classical.pipeline import ClassicalFaceRecPipeline
        from facerec_classical.config import Config

        config = Config.for_testing()
        config.data_dir = tmp_dataset_with_faces.parent

        pipeline = ClassicalFaceRecPipeline(config)
        metrics = pipeline.train(dataset_path=str(tmp_dataset_with_faces))

        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "reconstruction_error" in metrics

    def test_train_insufficient_data_raises(self, tmp_path):
        from facerec_classical.pipeline import ClassicalFaceRecPipeline
        from facerec_classical.config import Config

        config = Config.for_testing()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = ClassicalFaceRecPipeline(config)
        with pytest.raises(ValueError, match="Not enough"):
            pipeline.train(dataset_path=str(empty_dir))


class TestPipelineRecognize:
    """Tests for recognize() method."""

    def test_recognize_returns_results(self, tmp_dataset_with_faces, sample_bgr_image):
        from facerec_classical.pipeline import ClassicalFaceRecPipeline
        from facerec_classical.config import Config

        config = Config.for_testing()
        pipeline = ClassicalFaceRecPipeline(config)
        pipeline.train(dataset_path=str(tmp_dataset_with_faces))

        results = pipeline.recognize(sample_bgr_image)
        assert isinstance(results, list)
        assert len(results) >= 1

        for r in results:
            assert len(r.bbox) == 4
            assert isinstance(r.name, str)
            assert isinstance(r.distance, float)
