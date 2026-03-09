"""TDD tests for the face detection module.

RED phase — these tests define the expected interface and behaviour
of FaceDetector before any implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# STRUCTURAL / INTERFACE TESTS (no model download needed)
# ===================================================================

class TestFaceDetectorInterface:
    """Verify FaceDetector can be instantiated and has the right API."""

    def test_import(self):
        """FaceDetector is importable from facerec.detector."""
        from facerec.detector import FaceDetector
        assert FaceDetector is not None

    def test_instantiation_default(self):
        """FaceDetector can be created with default args (lazy model load)."""
        from facerec.detector import FaceDetector
        det = FaceDetector()
        assert det is not None

    def test_detection_dataclass_exists(self):
        """Detection result dataclass is importable."""
        from facerec.detector import Detection
        det = Detection(
            bbox=np.array([10, 20, 100, 150]),
            confidence=0.95,
            landmarks=np.zeros((5, 2)),
        )
        assert det.confidence == 0.95

    def test_detect_returns_list(self, sample_face_image):
        """detect() returns a list (possibly empty) on any image."""
        from facerec.detector import FaceDetector
        det = FaceDetector()
        results = det.detect(sample_face_image)
        assert isinstance(results, list)

    def test_detect_on_blank_image_returns_empty(self):
        """No faces should be found in a solid black image."""
        from facerec.detector import FaceDetector
        det = FaceDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(blank)
        assert results == []


# ===================================================================
# INTEGRATION TESTS (require model download — marked accordingly)
# ===================================================================

class TestFaceDetectorIntegration:
    """Integration tests that require actual model weights + real images.

    Run with: pytest -m integration
    """

    @pytest.mark.integration
    def test_detect_real_face_returns_at_least_one(self):
        """Given a real face image, detector should find ≥1 face."""
        from facerec.detector import FaceDetector
        from tests.conftest import FIXTURE_DIR

        # This test expects a real face image at tests/fixtures/face_sample.jpg
        img_path = FIXTURE_DIR / "face_sample.jpg"
        if not img_path.exists():
            pytest.skip("face_sample.jpg not found in fixtures — download LFW first")

        import cv2
        img = cv2.imread(str(img_path))
        det = FaceDetector()
        results = det.detect(img)

        assert len(results) >= 1
        for r in results:
            assert r.bbox.shape == (4,)
            assert 0 < r.confidence <= 1
            assert r.landmarks.shape == (5, 2)

    @pytest.mark.integration
    def test_bbox_coordinates_are_valid(self):
        """Bounding box coords should be non-negative and x1 < x2, y1 < y2."""
        from facerec.detector import FaceDetector
        from tests.conftest import FIXTURE_DIR

        img_path = FIXTURE_DIR / "face_sample.jpg"
        if not img_path.exists():
            pytest.skip("face_sample.jpg not found")

        import cv2
        img = cv2.imread(str(img_path))
        det = FaceDetector()
        results = det.detect(img)

        for r in results:
            x1, y1, x2, y2 = r.bbox
            assert x1 < x2, f"x1={x1} should be < x2={x2}"
            assert y1 < y2, f"y1={y1} should be < y2={y2}"
            assert x1 >= 0 and y1 >= 0
