"""Download LFW (Labeled Faces in the Wild) for local prototyping.

Adapted from Assignment2/scripts/download_lfw.py.

Supports three sources (tried in order):
  1. Kaggle API  — requires `kaggle` pip package + ~/.kaggle/kaggle.json
  2. sklearn     — built-in LFW fetcher, re-organises into per-person folders
  3. Direct URL  — original UMass mirror (fallback, may be unavailable)

Usage:
    python scripts/download_lfw.py [--output-dir data] [--source sklearn] [--min-faces 10]
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Source 1: Kaggle API
# ---------------------------------------------------------------------------

def _download_kaggle(output_dir: Path) -> Path:
    """Download LFW from Kaggle using the kaggle CLI/API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Kaggle source requires the 'kaggle' package.\n"
            "  pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json"
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset = "jessicali9530/lfw-dataset"
    download_path = output_dir / "kaggle_download"
    download_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading LFW from Kaggle ({dataset}) ...")
    api.dataset_download_files(dataset, path=str(download_path), unzip=True)

    lfw_root = _find_lfw_root(download_path)
    target = output_dir / "lfw"
    if lfw_root != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(lfw_root), str(target))

    if download_path.exists() and download_path != target:
        shutil.rmtree(download_path, ignore_errors=True)

    return target


def _find_lfw_root(search_dir: Path) -> Path:
    """Walk down until we find a directory full of person-named sub-dirs."""
    for root, dirs, files in os.walk(search_dir):
        root_path = Path(root)
        if len(dirs) > 100:
            return root_path
        for d in dirs:
            if d.lower() in ("lfw", "lfw-deepfunneled", "lfw_funneled"):
                candidate = root_path / d
                sub_count = sum(1 for _ in candidate.iterdir() if _.is_dir())
                if sub_count > 100:
                    return candidate
    return search_dir


# ---------------------------------------------------------------------------
# Source 2: scikit-learn (zero-config)
# ---------------------------------------------------------------------------

def _download_sklearn(output_dir: Path, min_faces: int = 10) -> Path:
    """Download LFW via sklearn and reorganize into per-person folders."""
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image
    import numpy as np

    print(f"Downloading LFW via scikit-learn (min_faces_per_person={min_faces}) ...")
    dataset = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,
        color=True,
    )

    target = output_dir / "lfw"
    target.mkdir(parents=True, exist_ok=True)

    names = dataset.target_names
    print(f"  Found {len(names)} identities with >= {min_faces} images")

    for i, (image_data, label) in enumerate(zip(dataset.images, dataset.target)):
        name = names[label]
        person_dir = target / name.replace(" ", "_")
        person_dir.mkdir(parents=True, exist_ok=True)

        img_uint8 = (image_data * 255).astype(np.uint8) if image_data.max() <= 1.0 else image_data.astype(np.uint8)
        img = Image.fromarray(img_uint8)
        img.save(person_dir / f"{name.replace(' ', '_')}_{i:04d}.jpg")

    count = sum(1 for _ in target.rglob("*.jpg"))
    print(f"  Saved {count} images to {target}")
    return target


# ---------------------------------------------------------------------------
# Source 3: Direct URL
# ---------------------------------------------------------------------------

DIRECT_URLS = [
    "https://ndownloader.figshare.com/articles/5976726/versions/1",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
]


def _download_direct(output_dir: Path) -> Path:
    """Download LFW from a direct URL."""
    import requests
    from tqdm import tqdm

    target = output_dir / "lfw"
    if target.exists():
        print(f"LFW already extracted at {target}")
        return target

    for url in DIRECT_URLS:
        archive = output_dir / "lfw_download"
        try:
            print(f"Trying {url} ...")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            ext = ".tgz" if ".tgz" in url else ".zip"
            archive = output_dir / f"lfw{ext}"

            with open(archive, "wb") as f:
                for chunk in tqdm(
                    resp.iter_content(chunk_size=8192),
                    total=total // 8192 if total else None,
                    unit="KB",
                    desc="Downloading",
                ):
                    f.write(chunk)

            print(f"Extracting to {output_dir} ...")
            if str(archive).endswith((".tgz", ".tar.gz")):
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(path=str(output_dir))
            elif str(archive).endswith(".zip"):
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(path=str(output_dir))

            if archive.exists():
                archive.unlink()

            lfw_root = _find_lfw_root(output_dir)
            if lfw_root != target and lfw_root.exists():
                shutil.move(str(lfw_root), str(target))

            return target

        except Exception as e:
            print(f"  Failed: {e}")
            if archive.exists():
                archive.unlink()
            continue

    raise RuntimeError("All download URLs failed. Try --source kaggle or --source sklearn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_lfw(output_dir: Path, source: str = "sklearn", min_faces: int = 10) -> Path:
    """Download LFW dataset using the specified source."""
    output_dir.mkdir(parents=True, exist_ok=True)

    target = output_dir / "lfw"
    if target.exists() and sum(1 for _ in target.iterdir() if _.is_dir()) > 10:
        print(f"LFW already exists at {target} — skipping download")
        return target

    if source == "kaggle":
        return _download_kaggle(output_dir)
    elif source == "sklearn":
        return _download_sklearn(output_dir, min_faces=min_faces)
    elif source == "direct":
        return _download_direct(output_dir)
    else:
        raise ValueError(f"Unknown source: {source}. Use kaggle, sklearn, or direct.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LFW dataset for Assignment 1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory to save the dataset (default: Assignment1/data)",
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "sklearn", "direct"],
        default="sklearn",
        help="Download source (default: sklearn — zero-config, most reliable)",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=10,
        help="Minimum images per person (sklearn source only, default: 10)",
    )
    args = parser.parse_args()
    lfw_path = download_lfw(args.output_dir, source=args.source, min_faces=args.min_faces)
    print(f"\nDone. LFW images at {lfw_path}")


if __name__ == "__main__":
    main()
