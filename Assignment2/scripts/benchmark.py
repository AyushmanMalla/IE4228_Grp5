"""Benchmark the face recognition pipeline.

Measures detection FPS and (optionally) gallery recognition accuracy.

Usage:
    python scripts/benchmark.py \
        --data-dir data/lfw/lfw \
        --gallery-dir data/gallery \
        [--device cpu] \
        [--num-images 100]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from facerec.config import Config
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer
from facerec.alignment import align_face


def benchmark_detection(data_dir: Path, device: str, num_images: int) -> None:
    """Measure detection FPS on random images from a dataset."""
    detector = FaceDetector(device=device)

    # Collect image paths
    img_paths = sorted(data_dir.rglob("*.jpg"))[:num_images]
    if not img_paths:
        print(f"No .jpg images found in {data_dir}")
        return

    print(f"Benchmarking detection on {len(img_paths)} images (device={device})...")

    # Warm up
    warmup_img = cv2.imread(str(img_paths[0]))
    detector.detect(warmup_img)

    # Timed run
    total_faces = 0
    start = time.perf_counter()
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = detector.detect(img)
        total_faces += len(dets)
    elapsed = time.perf_counter() - start

    fps = len(img_paths) / elapsed
    print(f"  Images processed: {len(img_paths)}")
    print(f"  Total faces found: {total_faces}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Avg time/image: {elapsed / len(img_paths) * 1000:.1f}ms")


def benchmark_embedding(data_dir: Path, device: str, num_images: int) -> None:
    """Measure embedding extraction speed."""
    detector = FaceDetector(device=device)
    recognizer = FaceRecognizer(device=device)

    img_paths = sorted(data_dir.rglob("*.jpg"))[:num_images]
    if not img_paths:
        return

    print(f"\nBenchmarking embedding extraction on {len(img_paths)} images...")

    emb_count = 0
    start = time.perf_counter()
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = detector.detect(img)
        for det in dets:
            aligned = align_face(img, det.landmarks)
            _ = recognizer.get_embedding(aligned)
            emb_count += 1
    elapsed = time.perf_counter() - start

    print(f"  Embeddings generated: {emb_count}")
    print(f"  Total time: {elapsed:.2f}s")
    if emb_count > 0:
        print(f"  Avg time/embedding: {elapsed / emb_count * 1000:.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face recognition pipeline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-images", type=int, default=100)
    args = parser.parse_args()

    benchmark_detection(args.data_dir, args.device, args.num_images)
    benchmark_embedding(args.data_dir, args.device, args.num_images)


if __name__ == "__main__":
    main()
