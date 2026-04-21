"""
===========================================================
 Smart Parking System - Dataset Preprocessor
 File   : preprocess.py
 Author : Smart Parking Team
 Purpose: Convert CNRPark (or any classification dataset)
          into YOLO detection format suitable for RT-DETRv2.

 CNRPark Dataset layout expected:
   data/raw/
     ├── BUSY/
     │   ├── img001.jpg
     │   └── ...
     └── FREE/
         ├── img001.jpg
         └── ...

 Output YOLO layout:
   data/
     ├── train/images/  & train/labels/
     ├── val/images/    & val/labels/
     └── test/images/   & test/labels/
===========================================================
"""

import os
import shutil
import random
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Paths resolved relative to this script's location (works from any CWD)
_HERE         = Path(__file__).resolve().parent
_BASE         = _HERE.parent
RAW_DIR       = _BASE / "data" / "raw"
PROCESSED_DIR = _BASE / "data" / "processed"
TRAIN_DIR     = _BASE / "data" / "train"
VAL_DIR       = _BASE / "data" / "val"
TEST_DIR      = _BASE / "data" / "test"

# Class mapping: 0 = FREE, 1 = OCCUPIED
CLASS_MAP = {"FREE": 0, "BUSY": 1}

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Image resize target (RT-DETR standard)
IMG_SIZE = 640

# Random seed for reproducible splits
RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create all required output directories."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Output directories ready.")


def make_yolo_label(class_id: int) -> str:
    """
    Generate a YOLO format label for a classification image.
    Since the whole image IS the parking slot, the bounding box
    covers the entire image:  class cx cy w h  →  class 0.5 0.5 1.0 1.0
    (all normalised to image size)
    """
    return f"{class_id} 0.500000 0.500000 1.000000 1.000000"


def resize_image(src_path: Path, dst_path: Path,
                 size: int = IMG_SIZE) -> bool:
    """
    Read, resize (preserving aspect ratio with padding), and save image.
    Returns True on success.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  [WARN] Cannot read image: {src_path}")
        return False

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Pad to square (letterbox style)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # grey padding
    pad_top  = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

    cv2.imwrite(str(dst_path), canvas)
    return True


def collect_samples() -> list:
    """
    Walk RAW_DIR and collect (image_path, class_id) pairs.
    Supports both flat structure and FREE/BUSY subfolder structure.
    """
    samples = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not RAW_DIR.exists():
        print(f"[ERROR] Raw data directory not found: {RAW_DIR}")
        return samples

    for class_name, class_id in CLASS_MAP.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"  [WARN] Class folder missing: {class_dir}")
            continue
        for img_file in class_dir.rglob("*"):
            if img_file.suffix.lower() in valid_exts:
                samples.append((img_file, class_id))

    print(f"[INFO] Found {len(samples)} total images "
          f"({sum(1 for _, c in samples if c==0)} FREE, "
          f"{sum(1 for _, c in samples if c==1)} OCCUPIED)")
    return samples


def split_samples(samples: list) -> Tuple[list, list, list]:
    """Shuffle and split into train / val / test."""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = samples[:n_train]
    val   = samples[n_train : n_train + n_val]
    test  = samples[n_train + n_val:]

    print(f"[INFO] Split → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def copy_split(samples: list, split_dir: Path, split_name: str) -> None:
    """Resize images and write YOLO labels for a single split."""
    img_out_dir = split_dir / "images"
    lbl_out_dir = split_dir / "labels"

    success = 0
    for i, (src_path, class_id) in enumerate(samples):
        stem = f"{split_name}_{i:06d}"
        dst_img  = img_out_dir / (stem + ".jpg")
        dst_lbl  = lbl_out_dir / (stem + ".txt")

        if resize_image(src_path, dst_img):
            # Write YOLO label
            dst_lbl.write_text(make_yolo_label(class_id))
            success += 1

    print(f"  [{split_name.upper()}] Processed {success}/{len(samples)} images.")


def generate_dataset_yaml() -> None:
    """Generate dataset.yaml required by Ultralytics training."""
    config = {
        "path"   : str(Path("../data").resolve()),
        "train"  : "train/images",
        "val"    : "val/images",
        "test"   : "test/images",
        "nc"     : len(CLASS_MAP),
        "names"  : {v: k for k, v in CLASS_MAP.items()},
    }
    yaml_path = PROCESSED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Dataset config saved: {yaml_path}")


def generate_stats_report(samples: list) -> None:
    """Print a quick summary of the dataset."""
    free_count = sum(1 for _, c in samples if c == 0)
    occ_count  = sum(1 for _, c in samples if c == 1)
    total      = len(samples)

    print("\n" + "=" * 50)
    print("  DATASET STATISTICS")
    print("=" * 50)
    print(f"  Total images    : {total}")
    print(f"  FREE  (class 0) : {free_count} ({100*free_count/max(total,1):.1f}%)")
    print(f"  BUSY  (class 1) : {occ_count}  ({100*occ_count/max(total,1):.1f}%)")
    print(f"  Train / Val / Test split: "
          f"{TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"  Output image size : {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 50 + "\n")


def run() -> None:
    """End-to-end preprocessing pipeline."""
    print("\n[START] Smart Parking Dataset Preprocessor")
    print("=" * 50)

    ensure_dirs()
    samples = collect_samples()

    if not samples:
        print("[ERROR] No samples found. Check your data/raw/ folder.")
        return

    generate_stats_report(samples)

    train, val, test = split_samples(samples)
    copy_split(train, TRAIN_DIR, "train")
    copy_split(val,   VAL_DIR,   "val")
    copy_split(test,  TEST_DIR,  "test")

    generate_dataset_yaml()

    print("\n[DONE] Preprocessing complete!")
    print(f"  Training data : {TRAIN_DIR}")
    print(f"  Dataset YAML  : {PROCESSED_DIR / 'dataset.yaml'}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
