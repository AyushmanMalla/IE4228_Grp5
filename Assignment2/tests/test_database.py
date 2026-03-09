"""TDD tests for the gallery database module.

RED phase — defines CRUD operations and similarity query contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_embedding(seed: int = 0) -> np.ndarray:
    """Helper: create a deterministic, unit-normalised 512-d vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


class TestGalleryDatabaseInterface:
    """Verify GalleryDatabase API."""

    def test_import(self):
        from facerec.database import GalleryDatabase
        assert GalleryDatabase is not None

    def test_create_empty(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        assert db.list_identities() == []

    def test_add_and_list_identity(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        embs = [_make_embedding(i) for i in range(3)]
        db.add_identity("alice", embs)
        assert "alice" in db.list_identities()

    def test_add_multiple_identities(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        db.add_identity("alice", [_make_embedding(0)])
        db.add_identity("bob", [_make_embedding(1)])
        names = db.list_identities()
        assert "alice" in names
        assert "bob" in names

    def test_remove_identity(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        db.add_identity("alice", [_make_embedding(0)])
        db.remove_identity("alice")
        assert "alice" not in db.list_identities()

    def test_remove_nonexistent_identity_raises(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        with pytest.raises(KeyError):
            db.remove_identity("nobody")


class TestGalleryQuery:
    """Verify similarity-based query behaviour."""

    def test_query_known_identity(self, tmp_gallery_dir):
        """Querying with a gallery member's own embedding should return their name."""
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        emb = _make_embedding(42)
        db.add_identity("charlie", [emb])

        name, score = db.query(emb, threshold=0.3)
        assert name == "charlie"
        assert score > 0.9  # should be ~1.0

    def test_query_unknown_identity(self, tmp_gallery_dir):
        """An embedding far from any gallery member should return 'Unknown'."""
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        db.add_identity("charlie", [_make_embedding(42)])

        # Very different embedding
        stranger = _make_embedding(999)
        name, score = db.query(stranger, threshold=0.9)
        assert name == "Unknown"

    def test_query_empty_database(self, tmp_gallery_dir):
        """Querying an empty database should return 'Unknown'."""
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        name, score = db.query(_make_embedding(0))
        assert name == "Unknown"


class TestGalleryPersistence:
    """Verify save/load round-trip."""

    def test_save_and_load_preserves_data(self, tmp_gallery_dir):
        from facerec.database import GalleryDatabase
        db = GalleryDatabase(tmp_gallery_dir)
        embs = [_make_embedding(i) for i in range(5)]
        db.add_identity("alice", embs[:3])
        db.add_identity("bob", embs[3:])
        db.save()

        # Load into a fresh instance
        db2 = GalleryDatabase(tmp_gallery_dir)
        db2.load()
        assert set(db2.list_identities()) == {"alice", "bob"}

        # Query still works after load
        name, score = db2.query(embs[0], threshold=0.3)
        assert name == "alice"
