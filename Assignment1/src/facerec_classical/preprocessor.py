"""Image pre-processing for classical face recognition.

Provides greyscale conversion, histogram equalisation, gamma
correction, and spatial normalisation (resize).
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


def resize_face(face: np.ndarray, target_size: tuple[int, int] = (100, 100)) -> np.ndarray:
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


def preprocess_face(
    image: np.ndarray,
    target_size: tuple[int, int] = (100, 100),
    gamma: float = 1.0,
) -> np.ndarray:
    """Full pre-processing pipeline: grayscale → histogram eq → gamma → resize.

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
        Preprocessed grayscale face, shape ``target_size``, dtype ``uint8``.
    """
    gray = to_grayscale(image)
    eq = equalize_histogram(gray)
    if gamma != 1.0:
        eq = gamma_correction(eq, gamma)
    return resize_face(eq, target_size)
