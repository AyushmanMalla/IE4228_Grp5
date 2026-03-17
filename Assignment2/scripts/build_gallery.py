"""Build a gallery database from a folder of face images.

Expects the folder structure (e.g. LFW):
    images_dir/
        PersonName/
            img1.jpg
            img2.jpg
        ...

By default, auto-selects the top-N identities with the most images.

Usage:
    python scripts/build_gallery.py \
        --images-dir data/lfw/lfw \
        --output-dir data/gallery \
        [--top-n 5] \
        [--device cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
 
SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

from facerec.alignment import align_face
from facerec.database import GalleryDatabase
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer


def _rank_identities(images_dir: Path, top_n: int | None) -> list[Path]:
    """Return person directories sorted by image count (descending).

    If top_n is set, only the top-N identities are returned.
    """
    person_dirs = [d for d in sorted(images_dir.iterdir()) if d.is_dir()]

    # Count images per person
    ranked = []
    for d in person_dirs:
        count = sum(len(list(d.glob(ext))) for ext in SUPPORTED_EXTENSIONS)
        if count > 0:
            ranked.append((d, count))

    ranked.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        ranked = ranked[:top_n]

    print(f"Selected {len(ranked)} identities:")
    for d, count in ranked:
        print(f"  {d.name:30s}  {count:>4d} images")
    print()

    return [d for d, _ in ranked]


def build_gallery(
    images_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    top_n: int | None = 5,
) -> None:
    """Detect, align, embed images and save as a gallery DB."""
    detector = FaceDetector(device=device)
    recognizer = FaceRecognizer(device=device)
    db = GalleryDatabase(output_dir)

    person_dirs = _rank_identities(images_dir, top_n)

    for person_dir in person_dirs:
        name = person_dir.name
        embeddings: list[np.ndarray] = []
        img_files = []
        for ext in SUPPORTED_EXTENSIONS:
            img_files.extend(person_dir.glob(ext))
        img_files.sort()

        for img_path in tqdm(img_files, desc=name, leave=True):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue

            detections = detector.detect(img)
            if not detections:
                continue

            # Use highest-confidence detection
            det = detections[0]
            aligned = align_face(img, det.landmarks, output_size=112)
            emb = recognizer.get_embedding(aligned)
            embeddings.append(emb)

        if embeddings:
            db.add_identity(name, embeddings)
            print(f"  → {name}: {len(embeddings)}/{len(img_files)} images embedded")
        else:
            print(f"  ⚠ {name}: no valid embeddings")

    db.save()
    print(f"\nGallery saved to {output_dir}")
    print(f"Identities: {db.list_identities()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gallery database from face images")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Root dir with person-named sub-folders of images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gallery",
    )
    parser.add_argument("--top-n", type=int, default=5,
                        help="Auto-select top N identities by image count (default: 5, use 0 for all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    top_n = args.top_n if args.top_n > 0 else None
    build_gallery(args.images_dir, args.output_dir, args.device, top_n)


if __name__ == "__main__":
    main()
