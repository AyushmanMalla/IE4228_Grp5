"""Face detection using OpenCV Haar Cascades (Viola-Jones).

Provides a thin wrapper around cv2.CascadeClassifier with tuneable
parameters and a structured Detection result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import cv2
import dlib


@dataclass
class Detection:
    """A single detected face."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    confidence: float


class FaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color).

        Returns
        -------
        list[Detection]
            Detected faces sorted by area (largest first).
        """
        pass


class DlibHOGFaceDetector(FaceDetector):
    """Classical 2005 HOG+SVM face detector via dlib (No neural networks)"""
    
    def __init__(self, upsample_num_times: int = 0) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.upsample_num_times = upsample_num_times

    def detect(self, image: np.ndarray) -> list[Detection]:
        # dlib expects RGB or Grayscale 8-bit
        if len(image.shape) == 3 and image.shape[2] == 3:
            # dlib expects RGB, cv2 reads BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image # Assume grayscale or already RGB

        rects = self.detector(rgb_image, self.upsample_num_times)
        
        detections = []
        h, w = image.shape[:2]
        for rect in rects:
            # dlib rect natively outputs [left, top, right, bottom]
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            
            # Secure boundary clipping
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Convert to (x, y, w, h) format for Detection
            bbox_x = x1
            bbox_y = y1
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            detections.append(Detection(
                bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                area=bbox_w * bbox_h,
                confidence=1.0  # Standard generic HOG confidence
            ))
            
        # Re-sort natively by bounding box area to lock the primary subject
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections


class HaarFaceDetector(FaceDetector):
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
        min_neighbors: int = 7,
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


class FaceAligner:
    """Align faces using dlib's 68-point landmark predictor.

    Computes a similarity transform (rotation + scale + translation)
    to pin eye centers to fixed target coordinates on a normalized crop.
    Falls back to simple resize if landmarks can't be detected.
    """

    # Target eye positions on a 200x200 crop
    LEFT_EYE_TARGET = (60, 70)
    RIGHT_EYE_TARGET = (140, 70)

    def __init__(self, predictor_path: str = "") -> None:
        from pathlib import Path

        if not predictor_path:
            # Look in data/ directory relative to project root
            project_root = Path(__file__).resolve().parents[2]
            predictor_path = str(
                project_root / "data" / "shape_predictor_68_face_landmarks.dat"
            )

        self._predictor = None
        if Path(predictor_path).exists():
            self._predictor = dlib.shape_predictor(predictor_path)
        else:
            print(
                f"Warning: dlib shape predictor not found at {predictor_path}. "
                f"Falling back to resize-only alignment."
            )

    def align(
        self,
        face_crop: np.ndarray,
        target_size: tuple[int, int] = (200, 200),
    ) -> np.ndarray:
        """Align a face crop using landmark-based similarity transform.

        Parameters
        ----------
        face_crop : np.ndarray
            Grayscale face crop.
        target_size : tuple[int, int]
            ``(width, height)`` of the output.

        Returns
        -------
        np.ndarray
            Aligned face crop of ``target_size``.
        """
        if face_crop.size == 0:
            return np.zeros(target_size[::-1], dtype=np.uint8)

        h, w = face_crop.shape[:2]

        if self._predictor is not None and h >= 30 and w >= 30:
            landmarks = self._detect_eye_centers(face_crop)
            if landmarks is not None:
                left_eye, right_eye = landmarks
                return self._similarity_transform(
                    face_crop, left_eye, right_eye, target_size
                )

        # Fallback: simple resize
        return cv2.resize(face_crop, target_size)

    def _detect_eye_centers(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect eye centers using dlib 68-point landmarks.

        Returns
        -------
        tuple or None
            ``(left_eye_center, right_eye_center)`` as float arrays,
            or ``None`` if detection fails.
        """
        rect = dlib.rectangle(0, 0, gray.shape[1], gray.shape[0])
        shape = self._predictor(gray, rect)

        if shape.num_parts < 68:
            return None

        # Left eye: landmarks 36-41, Right eye: landmarks 42-47
        left_eye = np.mean(
            [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)],
            axis=0,
        )
        right_eye = np.mean(
            [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)],
            axis=0,
        )

        return left_eye, right_eye

    def _similarity_transform(
        self,
        img: np.ndarray,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Apply similarity transform to pin eyes to target coordinates.

        Uses ``cv2.estimateAffinePartial2D`` which constrains to
        rotation + uniform scale + translation (4 DOF).
        """
        src_pts = np.float32([left_eye, right_eye])
        dst_pts = np.float32([self.LEFT_EYE_TARGET, self.RIGHT_EYE_TARGET])

        M, _ = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 1, 2),
            dst_pts.reshape(-1, 1, 2),
        )

        if M is None:
            return cv2.resize(img, target_size)

        aligned = cv2.warpAffine(
            img,
            M,
            target_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned
