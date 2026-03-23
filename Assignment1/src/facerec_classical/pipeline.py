"""End-to-end classical face recognition pipeline.

Orchestrates detection → preprocessing → PCA/LDA recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from facerec_classical.config import Config
from facerec_classical.database import FaceDatabase
from facerec_classical.detector import DlibHOGFaceDetector, FaceAligner
from facerec_classical.preprocessor import preprocess_face
from facerec_classical.recognizer import PCALDARecognizer


@dataclass
class RecognitionResult:
    """Result for a single detected face."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    name: str                         # identity name or "Unknown"
    distance: float                   # SED distance


class ClassicalFaceRecPipeline:
    """Full pipeline: image → detection → preprocessing → PCA/LDA → identity.

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._detector = DlibHOGFaceDetector()
        self._aligner = FaceAligner()
        self._recognizer = PCALDARecognizer(
            n_components_pca=self._config.n_components_pca,
            n_components_lda=self._config.n_components_lda,
            reconstruction_threshold=self._config.reconstruction_threshold,
            mahalanobis_threshold=self._config.mahalanobis_threshold,
        )

    def train(self, dataset_path: str | None = None) -> dict[str, Any]:
        """Train the recognition model from a folder-per-person dataset.

        Parameters
        ----------
        dataset_path : str | None
            Path to dataset. Defaults to ``config.data_dir / "lfw"``.

        Returns
        -------
        dict
            Training metrics from the recognizer.
        """
        if dataset_path is None:
            dataset_path = str(self._config.data_dir / "lfw")

        db = FaceDatabase(dataset_path)

        def _detect_and_preprocess(image_path: str) -> np.ndarray | None:
            """Load, detect face, preprocess."""
            import cv2
            from facerec_classical.detector import FaceAligner

            if not hasattr(self, "_aligner"):
                self._aligner = FaceAligner()

            img = cv2.imread(image_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = self._detector.detect(gray)

            if detections:
                x, y, w, h = detections[0].bbox
                face_crop = gray[y : y + h, x : x + w]
                face_crop = self._aligner.align(face_crop, target_size=self._config.target_size)
            else:
                face_crop = gray  # fallback: use whole image
                face_crop = self._aligner.align(face_crop, target_size=self._config.target_size)

            return preprocess_face(face_crop, target_size=self._config.target_size)

        X, y = db.load_dataset(preprocess_fn=_detect_and_preprocess)

        if len(X) < 2:
            raise ValueError("Not enough images to train.")

        return self._recognizer.fit(X, y)

    def recognize(self, image: np.ndarray) -> list[RecognitionResult]:
        """Run detection + recognition on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        list[RecognitionResult]
            One result per detected face.
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detect(gray)

        if not detections:
            # Fallback: treat whole image as a face
            h, w = gray.shape
            detections = [
                type(detections[0])(bbox=(0, 0, w, h), area=w * h)
                if detections
                else type("D", (), {"bbox": (0, 0, w, h), "area": w * h})()
            ]
            # Actually we need a proper Detection
            from facerec_classical.detector import Detection
            detections = [Detection(bbox=(0, 0, w, h), area=w * h, confidence=1.0)]

        results: list[RecognitionResult] = []

        for det in detections:
            x, y_coord, w, h = det.bbox
            face_crop = gray[y_coord : y_coord + h, x : x + w]
            face_preprocessed = preprocess_face(
                face_crop, target_size=self._config.target_size
            )
            face_vector = face_preprocessed.flatten()

            name, dist = self._recognizer.predict(face_vector)
            results.append(
                RecognitionResult(bbox=det.bbox, name=name, distance=dist)
            )

        return results
