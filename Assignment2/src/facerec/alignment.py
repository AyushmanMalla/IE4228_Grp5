"""Face alignment via 5-point landmark similarity transform.

Warps a detected face region into a canonical (112×112) crop suitable
for embedding extraction by ArcFace / AdaFace models.
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.transform import SimilarityTransform


# Standard ArcFace reference landmarks for a 112×112 crop.
# Source: insightface/utils/face_align.py
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Align a face using a 5-point similarity transform.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    landmarks : np.ndarray
        Five facial landmark points, shape ``(5, 2)``.
    output_size : int
        Side length of the square output crop (default 112).

    Returns
    -------
    np.ndarray
        Aligned face crop, shape ``(output_size, output_size, 3)``, uint8.

    Raises
    ------
    ValueError
        If ``landmarks`` does not have shape ``(5, 2)``.
    """
    if landmarks.shape != (5, 2):
        raise ValueError(
            f"landmarks must have shape (5, 2), got {landmarks.shape}"
        )

    # Scale reference landmarks to the desired output size
    scale = output_size / 112.0
    ref = ARCFACE_REF_LANDMARKS * scale

    # Estimate similarity transform (rotation + translation + uniform scale)
    tform = SimilarityTransform.from_estimate(landmarks, ref)

    # Warp the full image so the face lands on the reference positions
    M = tform.params[:2]  # 2×3 affine matrix
    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        borderValue=(0, 0, 0),
    )

    return aligned.astype(np.uint8)
