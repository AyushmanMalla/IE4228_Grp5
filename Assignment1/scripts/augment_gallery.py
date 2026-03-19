#!/usr/bin/env python3
"""Offline data augmentation script for Classical Face Recognition.

Expands the gallery by adding flipped, rotated, and brightened 
versions of original training images to improve PCA/LDA robustness.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def augment_image(img_path: Path) -> None:
    if img_path.name.startswith("aug_") or img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        return
        
    img = cv2.imread(str(img_path))
    if img is None:
        return
        
    base_name = img_path.name
    parent_dir = img_path.parent
    
    # 1. Flip
    flip_img = cv2.flip(img, 1)
    cv2.imwrite(str(parent_dir / f"aug_flip_{base_name}"), flip_img)
    
    # 2. Rotate +10 degrees
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rot10_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(parent_dir / f"aug_rot10_{base_name}"), rot10_img)
    
    # 3. Rotate -10 degrees
    M = cv2.getRotationMatrix2D(center, -10, 1.0)
    rotn10_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(parent_dir / f"aug_rot-10_{base_name}"), rotn10_img)
    
    # 4. Brightness +20%
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.2
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    br_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(parent_dir / f"aug_br20_{base_name}"), br_img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented training data")
    parser.add_argument(
        "--gallery-dir", type=Path, 
        default=Path(__file__).resolve().parents[1] / "data" / "team-photos",
        help="Path to the gallery directory"
    )
    args = parser.parse_args()
    
    if not args.gallery_dir.exists():
        print(f"Directory not found: {args.gallery_dir}")
        return

    print(f"Augmenting images in {args.gallery_dir}...")
    
    count = 0
    for root, _, files in os.walk(args.gallery_dir):
        for f in files:
            img_path = Path(root) / f
            # Only process original, non-hidden images
            if not f.startswith("aug_") and not f.startswith("."):
                augment_image(img_path)
                count += 1
                
    print(f"Augmentation complete. Processed {count} source images, yielding {count * 5} total images per identity.")


if __name__ == "__main__":
    main()
