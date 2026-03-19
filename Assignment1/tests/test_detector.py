"""TDD tests for the Haar Cascade face detector."""

from __future__ import annotations

import numpy as np
import pytest


class TestDlibHOGFaceDetectorInterface:
    """Verify DlibHOGFaceDetector can be instantiated and has the right API."""

    def test_import(self):
        """DlibHOGFaceDetector is importable from facerec_classical.detector."""
        from facerec_classical.detector import DlibHOGFaceDetector
        assert DlibHOGFaceDetector is not None

    def test_detection_dataclass_exists(self):
        """Detection result dataclass is importable and works."""
        from facerec_classical.detector import Detection
        det = Detection(bbox=(10, 20, 100, 150), area=15000, confidence=1.0)
        assert det.area == 15000
        assert det.confidence == 1.0
        assert det.bbox == (10, 20, 100, 150)

    def test_instantiation_default(self):
        """DlibHOGFaceDetector can be created with default args."""
        from facerec_classical.detector import DlibHOGFaceDetector
        det = DlibHOGFaceDetector()
        assert det is not None

    def test_detect_returns_list(self, sample_gray_image):
        """detect() returns a list on any image."""
        from facerec_classical.detector import DlibHOGFaceDetector
        det = DlibHOGFaceDetector()
        results = det.detect(sample_gray_image)
        assert isinstance(results, list)

    def test_detect_on_blank_returns_empty(self):
        """No faces should be found in a solid black image."""
        from facerec_classical.detector import DlibHOGFaceDetector
        det = DlibHOGFaceDetector()
        blank = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(blank)
        assert results == []

    def test_detection_bbox_format(self, sample_gray_image):
        """Each detection should have a 4-element bbox tuple."""
        from facerec_classical.detector import DlibHOGFaceDetector
        det = DlibHOGFaceDetector()
        results = det.detect(sample_gray_image)
        for r in results:
            assert len(r.bbox) == 4
            x, y, w, h = r.bbox
            assert w >= 0 and h >= 0

    def test_detections_sorted_by_area(self, sample_gray_image):
        """Detections should be sorted by area descending."""
        from facerec_classical.detector import DlibHOGFaceDetector
        det = DlibHOGFaceDetector()
        results = det.detect(sample_gray_image)
        if len(results) > 1:
            areas = [r.area for r in results]
            assert areas == sorted(areas, reverse=True)
