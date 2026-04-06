"""Image pre-processing for classical face recognition.

Provides greyscale conversion, CLAHE, gamma correction, spatial
normalisation (resize), HOG, and LBP feature extraction with fusion.
"""

from __future__ import annotations

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        BGR image of shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Grayscale image of shape ``(H, W)``, dtype ``uint8``.
    """
    import cv2

    if image.ndim == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Apply histogram equalisation to a grayscale image.

    .. deprecated:: Use :func:`clahe` instead for better local contrast.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image, dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Equalised image, same shape and dtype.
    """
    import cv2

    return cv2.equalizeHist(gray)


def clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on local tiles to preserve facial detail under partial shadow,
    unlike global histogram equalization which washes out features.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image, dtype ``uint8``.
    clip_limit : float
        Contrast amplification limit per tile. 2.0 is a safe default.
    tile_size : int
        Tile grid size in pixels.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image, same shape and dtype.
    """
    import cv2

    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
    )
    return clahe_obj.apply(gray)


def gamma_correction(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction to a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image, dtype ``uint8``.
    gamma : float
        Gamma value. > 1.0 darkens, < 1.0 brightens.

    Returns
    -------
    np.ndarray
        Corrected image, same shape and dtype.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    import cv2

    return cv2.LUT(gray, table)


def resize_face(face: np.ndarray, target_size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Resize a face crop to a fixed spatial size.

    Parameters
    ----------
    face : np.ndarray
        Face image (grayscale or colour).
    target_size : tuple[int, int]
        ``(width, height)`` of the output.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    import cv2

    return cv2.resize(face, target_size)


def compute_lbp_histogram(
    gray: np.ndarray,
    n_points: int = 8,
    radius: int = 1,
    grid_x: int = 5,
    grid_y: int = 5,
) -> np.ndarray:
    """Compute a spatially-gridded LBP histogram for texture description.

    Divides the image into a grid and computes a uniform LBP histogram
    per cell, then concatenates into a single 1D feature vector.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image, dtype ``uint8``.
    n_points : int
        Number of circularly symmetric neighbour points for LBP.
    radius : int
        Radius of circle for neighbour sampling.
    grid_x, grid_y : int
        Number of horizontal/vertical grid divisions.

    Returns
    -------
    np.ndarray
        Flattened 1D LBP histogram feature vector.
    """
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # For method='uniform' with P points, values range 0..P+1.
    # 0..P are the P+1 uniform patterns, P+1 is the non-uniform catch-all.
    # We keep all P+2 bins for completeness.
    n_bins = n_points + 2

    h, w = gray.shape
    cell_h = h // grid_y
    cell_w = w // grid_x

    histograms = []
    for gy in range(grid_y):
        for gx in range(grid_x):
            cell = lbp[
                gy * cell_h : (gy + 1) * cell_h,
                gx * cell_w : (gx + 1) * cell_w,
            ]
            hist, _ = np.histogram(
                cell.ravel(), bins=n_bins, range=(0, n_bins)
            )
            # L2-normalize per cell
            norm = np.linalg.norm(hist.astype(np.float64))
            if norm > 0:
                hist = hist.astype(np.float64) / norm
            else:
                hist = hist.astype(np.float64)
            histograms.append(hist)

    return np.concatenate(histograms)


def preprocess_face(
    image: np.ndarray,
    target_size: tuple[int, int] = (200, 200),
    gamma: float = 1.0,
) -> np.ndarray:
    """Full pre-processing pipeline: grayscale → CLAHE → gamma → resize → HOG + LBP.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale face crop.
    target_size : tuple[int, int]
        Output spatial size ``(width, height)``.
    gamma : float
        Gamma correction value (1.0 = skip).

    Returns
    -------
    np.ndarray
        Flattened 1D concatenated [HOG | LBP] feature vector.
    """
    from skimage.feature import hog

    gray = to_grayscale(image)
    eq = clahe(gray)
    if gamma != 1.0:
        eq = gamma_correction(eq, gamma)
    resized = resize_face(eq, target_size)

    # HOG features — shape/edge gradients (18,432 dims for 200×200)
    hog_features = hog(
        resized,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True,
    )

    # LBP features — micro-texture (250 dims for 5×5 grid, P=8)
    lbp_features = compute_lbp_histogram(resized)

    return np.concatenate([hog_features, lbp_features])
