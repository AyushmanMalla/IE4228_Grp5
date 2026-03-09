"""Shared test fixtures for the face recognition pipeline."""

from __future__ import annotations

import os
import tempfile
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
def sample_face_image() -> np.ndarray:
    """Create a synthetic 480×640 BGR image with a simple 'face-like' blob.

    This is NOT a real face — it's used for structural/interface tests.
    Integration tests that require real detection should use LFW images
    and be marked with @pytest.mark.integration.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a skin-toned ellipse in the centre to vaguely resemble a face
    import cv2
    centre = (320, 200)
    axes = (60, 80)
    cv2.ellipse(img, centre, axes, 0, 0, 360, (180, 200, 220), -1)
    # Two dark circles for "eyes"
    cv2.circle(img, (300, 180), 8, (40, 40, 40), -1)
    cv2.circle(img, (340, 180), 8, (40, 40, 40), -1)
    return img


@pytest.fixture
def aligned_face_112() -> np.ndarray:
    """A 112×112 BGR image simulating an aligned face crop."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def random_embedding() -> np.ndarray:
    """A unit-normalised 512-d embedding vector."""
    rng = np.random.RandomState(42)
    vec = rng.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def tmp_gallery_dir(tmp_path: Path) -> Path:
    """A temporary directory for gallery database tests."""
    gallery = tmp_path / "gallery_db"
    gallery.mkdir()
    return gallery


# ---------------------------------------------------------------------------
# Marks for slow / integration tests
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires model downloads / real images")
    config.addinivalue_line("markers", "slow: long-running test")
