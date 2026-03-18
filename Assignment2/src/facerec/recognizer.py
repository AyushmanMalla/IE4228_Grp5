"""Face recognition / embedding extraction module.

Uses insightface ArcFace (R100) by default.  Produces 512-d, L2-normalised
embeddings suitable for cosine-similarity matching.
"""

from __future__ import annotations

import numpy as np


class FaceRecognizer:
    """Extract facial embeddings and compute similarity.

    Parameters
    ----------
    model_name : str
        insightface model pack name (e.g. ``"buffalo_l"`` for ArcFace-R100).
    device : str
        ``"cpu"`` for local prototyping, ``"cuda"`` for NSCC A100.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._rec_model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._rec_model is not None:
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

        app = FaceAnalysis(
            name=self._model_name,
            providers=providers,
        )
        app.prepare(ctx_id=0 if self._device in ("cuda", "mps") else -1)

        # Extract the recognition model from the analysis app
        for model in app.models.values():
            if hasattr(model, "get_feat") or (
                hasattr(model, "taskname") and model.taskname == "recognition"
            ):
                self._rec_model = model
                break

        if self._rec_model is None:
            raise RuntimeError(
                f"No recognition model found in model pack '{self._model_name}'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Compute a 512-d unit-normalised embedding for an aligned face.

        Parameters
        ----------
        aligned_face : np.ndarray
            BGR image of shape ``(112, 112, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            512-d float32 embedding vector (L2-normalised).
        """
        self._ensure_loaded()
        assert self._rec_model is not None

        # insightface recognition models expect (112, 112, 3) BGR uint8
        embedding = self._rec_model.get_feat(aligned_face)

        # Flatten if batched
        if embedding.ndim == 2:
            embedding = embedding.flatten()

        embedding = embedding.astype(np.float32)

        # L2-normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Parameters
        ----------
        emb1, emb2 : np.ndarray
            512-d float32 vectors (ideally L2-normalised).

        Returns
        -------
        float
            Cosine similarity in ``[-1, 1]``.
        """
        dot = float(np.dot(emb1, emb2))
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)
