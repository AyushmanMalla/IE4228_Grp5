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
    """Uses Haar Cascades to detect eyes and align the face crop via affine rotation.
    
    This fulfills Strategy A: Geometry-based in-plane rotation to stabilize
    HOG and LBP descriptors.
    """
    def __init__(self, eye_cascade_path: str = "") -> None:
        import cv2
        if not eye_cascade_path:
            eye_cascade_path = str(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        self._cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self._cascade.empty():
            print(f"Warning: Failed to load eye cascade: {eye_cascade_path}")
            
    def align(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2
        import numpy as np
        
        if self._cascade.empty() or face_crop.size == 0:
            return face_crop
            
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20: 
            return face_crop
            
        # Eyes are generally in the top 60% of the face crop
        roi_gray = face_crop[0:int(h * 0.6), 0:w]
        
        eyes = self._cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
        if len(eyes) >= 2:
            # Sort by area (largest two = most likely eyes)
            eyes_list = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            # Sort strictly left-to-right
            eyes_list = sorted(eyes_list, key=lambda e: e[0])
            
            e1, e2 = eyes_list[0], eyes_list[1]
            c1 = (e1[0] + e1[2]//2, e1[1] + e1[3]//2)
            c2 = (e2[0] + e2[2]//2, e2[1] + e2[3]//2)
            
            dy = c2[1] - c1[1]
            dx = c2[0] - c1[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Avoid crazy false positive flips
            if abs(angle) < 45:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(
                    face_crop, M, (w, h), 
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                return aligned
                
        return face_crop
