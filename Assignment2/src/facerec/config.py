"""Central configuration for the face recognition pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return Assignment2/ directory regardless of cwd."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Pipeline configuration — all tuneable knobs live here."""

    # --- Paths ---
    project_root: Path = field(default_factory=_project_root)
    data_dir: Path = field(default=None)      # type: ignore[assignment]
    gallery_dir: Path = field(default=None)    # type: ignore[assignment]
    model_dir: Path = field(default=None)      # type: ignore[assignment]

    # --- Detection ---
    detector_model: str = "buffalo_l"  # insightface model pack (includes RetinaFace)
    detection_threshold: float = 0.5

    # --- Alignment ---
    aligned_face_size: int = 112  # standard for ArcFace / AdaFace

    # --- Recognition ---
    recognition_model: str = "buffalo_l"  # ArcFace-R100 from buffalo_l pack
    similarity_threshold: float = 0.35

    # --- Device ---
    device: str = "cpu"  # "cpu" for local, "cuda" for NSCC A100

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.gallery_dir is None:
            self.gallery_dir = self.data_dir / "gallery"
        if self.model_dir is None:
            self.model_dir = self.project_root / "models"

        # Auto-detect CUDA availability
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    @classmethod
    def for_testing(cls) -> Config:
        """Lightweight config for unit tests."""
        return cls(
            detection_threshold=0.3,
            similarity_threshold=0.25,
        )

    @classmethod
    def for_nscc(cls) -> Config:
        """Config tuned for NSCC Aspire2a A100 nodes."""
        return cls(
            device="cuda",
            detection_threshold=0.5,
            similarity_threshold=0.35,
        )
