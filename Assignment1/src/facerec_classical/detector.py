"""Face detection using OpenCV Haar Cascades (Viola-Jones).

Provides a thin wrapper around cv2.CascadeClassifier with tuneable
parameters and a structured Detection result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detected face."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area: int


class HaarFaceDetector:
    """Viola-Jones face detector backed by OpenCV Haar Cascades.

    Parameters
    ----------
    cascade_path : str
        Path to the Haar Cascade XML file.
    scale_factor : float
        Scale factor for multi-scale detection.
    min_neighbors : int
        Minimum neighbours for a positive detection.
    min_size : tuple[int, int]
        Minimum face size in pixels ``(w, h)``.
    """

    def __init__(
        self,
        cascade_path: str = "",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        import cv2

        if not cascade_path:
            cascade_path = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]

        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise FileNotFoundError(f"Failed to load cascade: {cascade_path}")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, gray_image: np.ndarray) -> list[Detection]:
        """Detect faces in a grayscale image.

        Parameters
        ----------
        gray_image : np.ndarray
            Single-channel grayscale image, dtype ``uint8``.

        Returns
        -------
        list[Detection]
            Detected faces sorted by area (largest first).
        """
        import cv2

        faces = self._cascade.detectMultiScale(
            gray_image,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )

        if len(faces) == 0:
            return []

        detections = [
            Detection(bbox=(int(x), int(y), int(w), int(h)), area=int(w * h))
            for (x, y, w, h) in faces
        ]
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
