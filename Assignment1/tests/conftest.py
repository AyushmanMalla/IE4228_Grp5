"""Shared test fixtures for the classical face recognition pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
FIXTURE_DIR = TESTS_DIR / "fixtures"


# ---------------------------------------------------------------------------
# Sample image generators
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """A 200×200 BGR image with a simple face-like blob for structural tests."""
    import cv2

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Skin-toned ellipse
    cv2.ellipse(img, (100, 90), (40, 55), 0, 0, 360, (180, 200, 220), -1)
    # Two dark circles for eyes
    cv2.circle(img, (85, 78), 5, (40, 40, 40), -1)
    cv2.circle(img, (115, 78), 5, (40, 40, 40), -1)
    return img


@pytest.fixture
def sample_gray_image(sample_bgr_image) -> np.ndarray:
    """Grayscale version of the sample image."""
    import cv2

    return cv2.cvtColor(sample_bgr_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def sample_face_100x100() -> np.ndarray:
    """A 100×100 grayscale face-like image for recognizer tests."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def tmp_gallery_dir(tmp_path: Path) -> Path:
    """A temporary directory for gallery database tests."""
    gallery = tmp_path / "gallery_db"
    gallery.mkdir()
    return gallery


@pytest.fixture
def tmp_dataset_with_faces(tmp_path: Path) -> Path:
    """Create a tiny dataset with 3 'people', 4 images each (10×10 grayscale)."""
    import cv2

    dataset = tmp_path / "tiny_dataset"
    rng = np.random.RandomState(123)

    for person_id in range(3):
        person_dir = dataset / f"Person_{person_id:02d}"
        person_dir.mkdir(parents=True)
        # Generate slightly different images per person
        base = rng.randint(50 * person_id, 50 * person_id + 50, (10, 10), dtype=np.uint8)
        for img_id in range(4):
            noise = rng.randint(0, 10, (10, 10), dtype=np.uint8)
            img = np.clip(base.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
            cv2.imwrite(str(person_dir / f"img_{img_id:02d}.jpg"), img)

    return dataset


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires dataset downloads / real images")
    config.addinivalue_line("markers", "slow: long-running test")
