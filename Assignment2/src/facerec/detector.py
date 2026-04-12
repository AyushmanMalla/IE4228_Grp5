"""Face detection module using insightface.

Supports multiple backends: SCRFD-10GF (accuracy, buffalo_l) and
SCRFD-500MF (speed, buffalo_sc).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detected face."""

    bbox: np.ndarray       # shape (4,) — [x1, y1, x2, y2]
    confidence: float      # detection confidence ∈ (0, 1]
    landmarks: np.ndarray  # shape (5, 2) — 5 facial landmarks


class FaceDetector:
    """Face detector backed by insightface model zoo.

    Parameters
    ----------
    model_name : str
        insightface model pack name.  ``"buffalo_l"`` uses SCRFD-10GF
        (10 GFLOPs).  ``"buffalo_sc"`` uses SCRFD-500MF for speed.
    device : str
        ``"cpu"`` for local prototyping, ``"cuda"`` for NSCC A100.
    det_thresh : float
        Minimum confidence to keep a detection.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cpu",
        det_thresh: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._det_thresh = det_thresh
        self._app = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download + load the insightface model pack on first use."""
        if self._app is not None:
            return

        import insightface
        from insightface.app import FaceAnalysis

        providers: list[str]
        if self._device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self._device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(
            name=self._model_name,
            providers=providers,
        )
        self._app.prepare(ctx_id=0 if self._device in ("cuda", "mps") else -1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image of shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        list[Detection]
            Detected faces sorted by descending confidence.
        """
        self._ensure_loaded()
        assert self._app is not None

        raw_faces = self._app.get(image)

        detections: list[Detection] = []
        for face in raw_faces:
            conf = float(face.det_score)
            if conf < self._det_thresh:
                continue

            detections.append(
                Detection(
                    bbox=face.bbox.astype(np.float32),  # (4,)
                    confidence=conf,
                    landmarks=face.kps if face.kps is not None else np.zeros((5, 2)),
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
