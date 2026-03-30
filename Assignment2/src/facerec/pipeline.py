"""End-to-end face recognition pipeline.

Orchestrates detection → alignment → embedding → gallery query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from facerec.alignment import align_face
from facerec.config import Config
from facerec.database import GalleryDatabase
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer

import time
import cv2


@dataclass
class RecognitionResult:
    """Result for a single detected face."""

    bbox: np.ndarray         # (4,) — [x1, y1, x2, y2]
    name: str                # identity name or "Unknown"
    confidence: float        # detection confidence
    embedding: np.ndarray    # (512,) — face embedding


class FaceRecognitionPipeline:
    """Full pipeline: image → detections → aligned crops → embeddings → IDs.

    Parameters
    ----------
    config : Config
        Pipeline configuration (thresholds, model names, paths).
    gallery : GalleryDatabase | None
        Pre-loaded gallery.  If ``None``, a new empty gallery is created.
    """

    def __init__(self, config: Config, gallery: GalleryDatabase | None = None) -> None:
        self._config = config
        self._detector = FaceDetector(
            model_name=config.detector_model,
            device=config.device,
            det_thresh=config.detection_threshold,
        )
        self._recognizer = FaceRecognizer(
            model_name=config.recognition_model,
            device=config.device,
        )
        if gallery is not None:
            self._gallery = gallery
        else:
            self._gallery = GalleryDatabase(config.gallery_dir)
            gallery_json = config.gallery_dir / "gallery.json"
            if gallery_json.exists():
                self._gallery.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image: np.ndarray) -> list[RecognitionResult]:
        """Run the full pipeline on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        list[RecognitionResult]
            One result per detected face, sorted by confidence (desc).
        """
        detections = self._detector.detect(image)
        results: list[RecognitionResult] = []

        for det in detections:
            # Align using 5 landmarks
            aligned = align_face(
                image,
                det.landmarks,
                output_size=self._config.aligned_face_size,
            )

            # Extract embedding
            embedding = self._recognizer.get_embedding(aligned)

            # Query gallery
            name, _score = self._gallery.query(
                embedding,
                threshold=self._config.similarity_threshold,
            )

            results.append(
                RecognitionResult(
                    bbox=det.bbox,
                    name=name,
                    confidence=det.confidence,
                    embedding=embedding,
                )
            )

        return results

    def register_new_identity(
        self,
        frame_bgr: np.ndarray,
        detection: Any,
        name: str,
        scale: float = 1.0
    ) -> None:
        """Handles the full backend logic of registering a new face live."""
        # 1. Save the actual photo to the team_photos folder
        project_root = Path(__file__).resolve().parents[2]
        photos_dir = project_root / "data" / "team_photos" / name
        photos_dir.mkdir(parents=True, exist_ok=True)
        filename = photos_dir / f"live_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame_bgr)

        # 2. Extract the math embedding using the pipeline's own recognizer
        lms = detection.landmarks / scale
        crop = align_face(frame_bgr, lms, output_size=self._config.aligned_face_size)
        emb = self._recognizer.get_embedding(crop)
        
        # 3. Save to the pipeline's database
        self._gallery.add_identity(name, [emb])
        self._gallery.save()