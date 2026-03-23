"""Central configuration for the classical face recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return Assignment1/ directory regardless of cwd."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Pipeline configuration — all tuneable knobs live here."""

    # --- Paths ---
    project_root: Path = field(default_factory=_project_root)
    data_dir: Path = field(default=None)       # type: ignore[assignment]
    gallery_dir: Path = field(default=None)     # type: ignore[assignment]
    cascade_path: str = ""

    # --- Pre-processing ---
    target_size: tuple[int, int] = (100, 100)

    # --- PCA ---
    n_components_pca: int = 50
    pca_whiten: bool = True

    # --- LDA ---
    n_components_lda: str | int = "auto"  # "auto" = min(n_classes-1, pca_dims)

    # --- Recognition    # Thresholds
    sed_threshold: float = 0.30

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.gallery_dir is None:
            self.gallery_dir = self.data_dir / "gallery"
        if not self.cascade_path:
            import cv2
            self.cascade_path = str(
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
            )

    @classmethod
    def for_testing(cls) -> "Config":
        """Lightweight config for unit tests."""
        return cls(
            n_components_pca=10,
            sed_threshold=100.0,
        )
