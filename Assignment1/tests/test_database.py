"""TDD tests for the face database / gallery module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestFaceDatabaseInterface:
    """Verify FaceDatabase API."""

    def test_import(self):
        from facerec_classical.database import FaceDatabase
        assert FaceDatabase is not None

    def test_instantiation(self, tmp_gallery_dir):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_gallery_dir)
        assert db is not None


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_load_from_folder_structure(self, tmp_dataset_with_faces):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_dataset_with_faces)
        X, y = db.load_dataset()

        assert len(X) == 12  # 3 people × 4 images
        assert len(y) == 12
        assert len(np.unique(y)) == 3

    def test_load_nonexistent_dir_returns_empty(self, tmp_path):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_path / "nonexistent")
        X, y = db.load_dataset()
        assert len(X) == 0
        assert len(y) == 0

    def test_load_with_preprocess_fn(self, tmp_dataset_with_faces):
        from facerec_classical.database import FaceDatabase

        def dummy_preprocess(path: str) -> np.ndarray:
            return np.ones(25, dtype=np.uint8)  # 5×5 flattened

        db = FaceDatabase(tmp_dataset_with_faces)
        X, y = db.load_dataset(preprocess_fn=dummy_preprocess)
        assert X.shape[1] == 25


class TestGetLabels:
    """Tests for get_labels()."""

    def test_returns_sorted_names(self, tmp_dataset_with_faces):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_dataset_with_faces)
        labels = db.get_labels()
        assert labels == ["Person_00", "Person_01", "Person_02"]

    def test_empty_dir_returns_empty(self, tmp_gallery_dir):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_gallery_dir)
        assert db.get_labels() == []


class TestAddRemoveIdentity:
    """Tests for modular gallery add/remove."""

    def test_add_identity_creates_folder(self, tmp_gallery_dir, tmp_path):
        from facerec_classical.database import FaceDatabase
        import cv2

        # Create a temp image
        img_path = tmp_path / "face.jpg"
        cv2.imwrite(str(img_path), np.zeros((10, 10), dtype=np.uint8))

        db = FaceDatabase(tmp_gallery_dir)
        db.add_identity("Alice", [str(img_path)])

        assert "Alice" in db.get_labels()
        assert (tmp_gallery_dir / "Alice" / "face.jpg").exists()

    def test_add_identity_empty_raises(self, tmp_gallery_dir):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_gallery_dir)
        with pytest.raises(ValueError, match="at least one"):
            db.add_identity("Bob", [])

    def test_remove_identity(self, tmp_gallery_dir, tmp_path):
        from facerec_classical.database import FaceDatabase
        import cv2

        img_path = tmp_path / "face.jpg"
        cv2.imwrite(str(img_path), np.zeros((10, 10), dtype=np.uint8))

        db = FaceDatabase(tmp_gallery_dir)
        db.add_identity("Charlie", [str(img_path)])
        assert "Charlie" in db.get_labels()

        db.remove_identity("Charlie")
        assert "Charlie" not in db.get_labels()

    def test_remove_nonexistent_raises(self, tmp_gallery_dir):
        from facerec_classical.database import FaceDatabase
        db = FaceDatabase(tmp_gallery_dir)
        with pytest.raises(KeyError, match="not found"):
            db.remove_identity("Nonexistent")
