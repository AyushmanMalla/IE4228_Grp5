"""TDD tests for the pre-processing module."""

from __future__ import annotations

import numpy as np
import pytest


class TestToGrayscale:
    """Tests for to_grayscale()."""

    def test_bgr_to_gray_reduces_channels(self, sample_bgr_image):
        from facerec_classical.preprocessor import to_grayscale
        gray = to_grayscale(sample_bgr_image)
        assert gray.ndim == 2
        assert gray.shape == sample_bgr_image.shape[:2]

    def test_already_gray_is_noop(self, sample_gray_image):
        from facerec_classical.preprocessor import to_grayscale
        result = to_grayscale(sample_gray_image)
        np.testing.assert_array_equal(result, sample_gray_image)


class TestEqualizeHistogram:
    """Tests for equalize_histogram()."""

    def test_output_shape_matches_input(self, sample_gray_image):
        from facerec_classical.preprocessor import equalize_histogram
        eq = equalize_histogram(sample_gray_image)
        assert eq.shape == sample_gray_image.shape
        assert eq.dtype == np.uint8

    def test_changes_pixel_distribution(self):
        from facerec_classical.preprocessor import equalize_histogram
        # Low-contrast image
        low_contrast = np.full((50, 50), 128, dtype=np.uint8)
        low_contrast[:25, :25] = 130
        eq = equalize_histogram(low_contrast)
        # After equalisation, the range should be wider
        assert eq.max() > low_contrast.max() or eq.min() < low_contrast.min()


class TestGammaCorrection:
    """Tests for gamma_correction()."""

    def test_gamma_1_is_near_identity(self, sample_gray_image):
        from facerec_classical.preprocessor import gamma_correction
        result = gamma_correction(sample_gray_image, gamma=1.0)
        assert result.shape == sample_gray_image.shape
        # gamma=1.0 should be very close to identity
        np.testing.assert_allclose(result.astype(float), sample_gray_image.astype(float), atol=1)

    def test_gamma_lt1_darkens(self):
        from facerec_classical.preprocessor import gamma_correction
        img = np.full((10, 10), 128, dtype=np.uint8)
        darkened = gamma_correction(img, gamma=0.5)
        assert darkened.mean() < img.mean()

    def test_gamma_gt1_brightens(self):
        from facerec_classical.preprocessor import gamma_correction
        img = np.full((10, 10), 128, dtype=np.uint8)
        brightened = gamma_correction(img, gamma=2.0)
        assert brightened.mean() > img.mean()


class TestResizeFace:
    """Tests for resize_face()."""

    def test_output_matches_target_size(self, sample_gray_image):
        from facerec_classical.preprocessor import resize_face
        resized = resize_face(sample_gray_image, target_size=(100, 100))
        assert resized.shape == (100, 100)

    def test_different_target_sizes(self):
        from facerec_classical.preprocessor import resize_face
        img = np.zeros((50, 80), dtype=np.uint8)
        for size in [(64, 64), (100, 100), (32, 32)]:
            resized = resize_face(img, target_size=size)
            assert resized.shape == (size[1], size[0])  # cv2.resize returns (w, h)


class TestPreprocessFace:
    """Tests for the convenience preprocess_face() pipeline."""

    def test_full_pipeline_output_shape(self, sample_bgr_image):
        from facerec_classical.preprocessor import preprocess_face
        result = preprocess_face(sample_bgr_image, target_size=(100, 100))
        assert result.shape == (3872,)
        assert result.dtype in (np.float32, np.float64)

    def test_full_pipeline_with_gamma(self, sample_bgr_image):
        from facerec_classical.preprocessor import preprocess_face
        result = preprocess_face(sample_bgr_image, target_size=(100, 100), gamma=1.5)
        assert result.shape == (3872,)
