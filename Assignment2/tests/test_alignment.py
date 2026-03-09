"""TDD tests for the face alignment module.

RED phase — defines the expected interface for align_face().
"""

from __future__ import annotations

import numpy as np
import pytest


class TestAlignFace:
    """Verify the alignment function contract."""

    def test_import(self):
        """align_face is importable from facerec.alignment."""
        from facerec.alignment import align_face
        assert callable(align_face)

    def test_output_shape_default(self, sample_face_image):
        """Aligned face output should be (112, 112, 3) by default."""
        from facerec.alignment import align_face

        # Provide 5 fake landmark points within the image bounds
        landmarks = np.array([
            [300, 180],  # left eye
            [340, 180],  # right eye
            [320, 210],  # nose
            [305, 240],  # left mouth
            [335, 240],  # right mouth
        ], dtype=np.float32)

        aligned = align_face(sample_face_image, landmarks, output_size=112)
        assert aligned.shape == (112, 112, 3)

    def test_output_dtype_is_uint8(self, sample_face_image):
        """Aligned face should be uint8 (pixel values 0-255)."""
        from facerec.alignment import align_face

        landmarks = np.array([
            [300, 180], [340, 180], [320, 210], [305, 240], [335, 240],
        ], dtype=np.float32)

        aligned = align_face(sample_face_image, landmarks, output_size=112)
        assert aligned.dtype == np.uint8

    def test_custom_output_size(self, sample_face_image):
        """align_face should respect a custom output_size parameter."""
        from facerec.alignment import align_face

        landmarks = np.array([
            [300, 180], [340, 180], [320, 210], [305, 240], [335, 240],
        ], dtype=np.float32)

        aligned = align_face(sample_face_image, landmarks, output_size=224)
        assert aligned.shape == (224, 224, 3)

    def test_landmarks_must_be_5x2(self, sample_face_image):
        """Should raise ValueError if landmarks shape is not (5, 2)."""
        from facerec.alignment import align_face

        bad_landmarks = np.array([[100, 200], [150, 200]], dtype=np.float32)

        with pytest.raises(ValueError, match="5"):
            align_face(sample_face_image, bad_landmarks)
