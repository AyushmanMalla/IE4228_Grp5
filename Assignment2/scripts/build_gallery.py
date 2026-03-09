"""Build a gallery database from a folder of face images.

Expects the folder structure:
    gallery_images/
        Alice/
            img1.jpg
            img2.jpg
        Bob/
            img1.jpg
            ...

Usage:
    python scripts/build_gallery.py \
        --images-dir data/gallery_images \
        --output-dir data/gallery \
        [--device cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from facerec.alignment import align_face
from facerec.config import Config
from facerec.database import GalleryDatabase
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer


def build_gallery(images_dir: Path, output_dir: Path, device: str = "cpu") -> None:
    """Detect, align, embed all images and save as a gallery DB."""
    detector = FaceDetector(device=device)
    recognizer = FaceRecognizer(device=device)
    db = GalleryDatabase(output_dir)

    for person_dir in sorted(images_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        embeddings: list[np.ndarray] = []
        img_files = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))

        print(f"Processing {name}: {len(img_files)} images")

        for img_path in tqdm(img_files, desc=name, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue

            detections = detector.detect(img)
            if not detections:
                print(f"  Warning: no face found in {img_path.name}")
                continue

            # Use highest-confidence detection
            det = detections[0]
            aligned = align_face(img, det.landmarks, output_size=112)
            emb = recognizer.get_embedding(aligned)
            embeddings.append(emb)

        if embeddings:
            db.add_identity(name, embeddings)
            print(f"  → Added {name} with {len(embeddings)} embeddings")
        else:
            print(f"  ⚠ No valid embeddings for {name}")

    db.save()
    print(f"\nGallery saved to {output_dir}")
    print(f"Identities: {db.list_identities()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gallery database")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gallery",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    build_gallery(args.images_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
