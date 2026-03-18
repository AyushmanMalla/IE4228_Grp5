"""Gallery database for classical face recognition.

Loads face images from a folder-per-person directory structure and
provides modular add/remove identity support for swapping between
LFW prototyping and teammate gallery.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np


class FaceDatabase:
    """On-disk face image gallery with folder-per-identity layout.

    Parameters
    ----------
    dataset_path : str | Path
        Root directory containing one subfolder per identity, each
        containing face image files.
    """

    def __init__(self, dataset_path: str | Path) -> None:
        self._path = Path(dataset_path)

    def load_dataset(
        self,
        preprocess_fn=None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load all images from the dataset directory.

        Parameters
        ----------
        preprocess_fn : callable, optional
            Function ``(image_path: str) -> np.ndarray | None``.
            If ``None``, images are loaded as-is and flattened.
        extensions : tuple[str, ...]
            Image file extensions to include.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X, y)`` where ``X`` has shape ``(n_samples, n_features)``
            and ``y`` has shape ``(n_samples,)`` with string labels.
        """
        if not self._path.exists():
            return np.array([]), np.array([])

        X: list[np.ndarray] = []
        y: list[str] = []

        for person_dir in sorted(self._path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name

            for img_file in sorted(person_dir.iterdir()):
                if img_file.suffix.lower() not in extensions:
                    continue

                if preprocess_fn is not None:
                    processed = preprocess_fn(str(img_file))
                    if processed is not None:
                        X.append(processed.flatten())
                        y.append(person_name)
                else:
                    import cv2
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        X.append(img.flatten())
                        y.append(person_name)

        if not X:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def get_labels(self) -> list[str]:
        """Return sorted list of identity names in the gallery."""
        if not self._path.exists():
            return []

        return sorted(
            d.name for d in self._path.iterdir() if d.is_dir()
        )

    def add_identity(self, name: str, image_paths: list[str]) -> None:
        """Add a new identity by copying images into the gallery.

        Parameters
        ----------
        name : str
            Identity name (used as folder name).
        image_paths : list[str]
            Absolute paths to face images to copy.
        """
        if not image_paths:
            raise ValueError("Must provide at least one image path")

        person_dir = self._path / name
        person_dir.mkdir(parents=True, exist_ok=True)

        for src in image_paths:
            src_path = Path(src)
            if src_path.exists():
                shutil.copy2(str(src_path), str(person_dir / src_path.name))

    def remove_identity(self, name: str) -> None:
        """Remove an identity and all its images.

        Parameters
        ----------
        name : str
            Identity name to remove.

        Raises
        ------
        KeyError
            If identity doesn't exist.
        """
        person_dir = self._path / name
        if not person_dir.exists():
            raise KeyError(f"Identity '{name}' not found in gallery")
        shutil.rmtree(str(person_dir))
