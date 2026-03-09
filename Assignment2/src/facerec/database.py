"""Gallery database for identity management and similarity queries.

Stores per-identity embeddings as ``.npz`` files and metadata in
``gallery.json``.  Query uses cosine similarity against all stored
embeddings and returns the best match or ``"Unknown"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class GalleryDatabase:
    """On-disk gallery of known-identity face embeddings.

    Parameters
    ----------
    db_path : str | Path
        Directory where gallery data is stored.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.mkdir(parents=True, exist_ok=True)

        # In-memory store: name → list of 512-d embeddings
        self._identities: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_identity(self, name: str, embeddings: list[np.ndarray]) -> None:
        """Add (or overwrite) an identity with one or more embeddings."""
        if not embeddings:
            raise ValueError("Must provide at least one embedding")
        self._identities[name] = [e.astype(np.float32) for e in embeddings]

    def remove_identity(self, name: str) -> None:
        """Remove an identity by name.  Raises ``KeyError`` if not found."""
        if name not in self._identities:
            raise KeyError(f"Identity '{name}' not in gallery")
        del self._identities[name]

    def list_identities(self) -> list[str]:
        """Return sorted list of identity names."""
        return sorted(self._identities.keys())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: np.ndarray,
        threshold: float = 0.35,
    ) -> tuple[str, float]:
        """Find the closest matching identity.

        Parameters
        ----------
        embedding : np.ndarray
            512-d query embedding.
        threshold : float
            Minimum cosine similarity to accept a match.

        Returns
        -------
        tuple[str, float]
            ``(name, score)`` of best match, or ``("Unknown", best_score)``
            if below threshold.
        """
        if not self._identities:
            return ("Unknown", 0.0)

        embedding = embedding.astype(np.float32)
        e_norm = np.linalg.norm(embedding)
        if e_norm > 0:
            embedding = embedding / e_norm

        best_name = "Unknown"
        best_score = -1.0

        for name, embs in self._identities.items():
            for stored in embs:
                s_norm = np.linalg.norm(stored)
                if s_norm > 0:
                    stored_normed = stored / s_norm
                else:
                    stored_normed = stored
                score = float(np.dot(embedding, stored_normed))
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < threshold:
            return ("Unknown", best_score)

        return (best_name, best_score)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist gallery to disk (``gallery.json`` + ``.npz`` per identity)."""
        meta: dict[str, int] = {}  # name → embedding count

        for name, embs in self._identities.items():
            arr = np.stack(embs)  # (N, 512)
            np.savez_compressed(self._path / f"{name}.npz", embeddings=arr)
            meta[name] = len(embs)

        with open(self._path / "gallery.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self) -> None:
        """Load gallery from disk."""
        meta_path = self._path / "gallery.json"
        if not meta_path.exists():
            return

        with open(meta_path) as f:
            meta: dict[str, int] = json.load(f)

        self._identities = {}
        for name in meta:
            npz_path = self._path / f"{name}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                arr = data["embeddings"]  # (N, 512)
                self._identities[name] = [
                    arr[i].astype(np.float32) for i in range(arr.shape[0])
                ]
