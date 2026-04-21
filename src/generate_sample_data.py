# -*- coding: utf-8 -*-
"""
===========================================================
 Smart Parking System - Sample Data Generator
 File   : generate_sample_data.py
 Purpose: Creates synthetic parking lot images for testing
          the full pipeline without a real camera/dataset.
          Generates:
            - data/raw/parking_lot_frame.png  (reference image)
            - data/raw/FREE/  (200 sample free slot images)
            - data/raw/BUSY/  (200 sample busy slot images)
            - data/processed/parking_positions.pkl (pre-set ROIs)
===========================================================
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
PROC_DIR= BASE / "data" / "processed"

SLOT_W = 107
SLOT_H = 48

# ──────────────────────────────────────────────────────────────
# 1. PARKING LOT REFERENCE IMAGE
# ──────────────────────────────────────────────────────────────
def draw_parking_lot(width=1280, height=720) -> np.ndarray:
    """Synthesise a top-down parking lot image."""
    img = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)  # grey asphalt

    # Ground texture (random noise)
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    img   = cv2.add(img, noise)

    # Draw lane markings (white lines)
    for y in range(60, height - 60, SLOT_H + 12):
        cv2.line(img, (50, y), (width - 50, y), (220, 220, 220), 2)
    for y in range(60 + SLOT_H, height - 60, SLOT_H + 12):
        cv2.line(img, (50, y), (width - 50, y), (220, 220, 220), 2)

    # Draw vertical slot dividers
    num_cols = (width - 100) // (SLOT_W + 10)
    for row_start_y in range(60, height - 60, SLOT_H + 12):
        for col in range(num_cols + 1):
            x = 50 + col * (SLOT_W + 10)
            cv2.line(img, (x, row_start_y), (x, row_start_y + SLOT_H),
                     (200, 200, 200), 1)

    # Some random "cars" (darker rectangles)
    np.random.seed(42)
    for row_start_y in range(60, height - 60, SLOT_H + 12):
        for col in range(num_cols):
            if np.random.rand() > 0.45:   # 55% chance of a car
                x = 50 + col * (SLOT_W + 10) + 4
                y = row_start_y + 4
                car_color = tuple(int(c) for c in np.random.randint(30, 100, 3))
                cv2.rectangle(img, (x, y),
                              (x + SLOT_W - 8, y + SLOT_H - 8),
                              car_color, -1)
                # Windshield glint
                cv2.rectangle(img,
                              (x + 8, y + 6),
                              (x + SLOT_W - 16, y + 14),
                              (160, 160, 200), -1)

    # Axis labels
    cv2.putText(img, "SMART PARKING LOT — Synthetic Reference",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ──────────────────────────────────────────────────────────────
# 2. SLOT SAMPLE IMAGES (FREE & BUSY)
# ──────────────────────────────────────────────────────────────
def make_free_slot(idx: int) -> np.ndarray:
    """Generate a synthetic empty parking slot (64×64)."""
    img = np.full((64, 64, 3), np.random.randint(70, 100), dtype=np.uint8)
    # Asphalt texture
    noise = np.random.randint(0, 25, (64, 64, 3), dtype=np.uint8)
    img   = cv2.add(img, noise)
    # Lane-marking lines at edges
    cv2.line(img, (0, 0), (63, 0), (200, 200, 200), 2)
    cv2.line(img, (0, 63), (63, 63), (200, 200, 200), 2)
    cv2.line(img, (0, 0), (0, 63), (200, 200, 200), 1)
    cv2.line(img, (63, 0), (63, 63), (200, 200, 200), 1)
    # Slight brightness variation per image
    brightness = np.random.randint(-15, 15)
    img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return img


def make_busy_slot(idx: int) -> np.ndarray:
    """Generate a synthetic occupied parking slot with a 'car'."""
    img = np.full((64, 64, 3), np.random.randint(60, 90), dtype=np.uint8)
    noise = np.random.randint(0, 20, (64, 64, 3), dtype=np.uint8)
    img   = cv2.add(img, noise)
    # Car body (random dark colour)
    car_col = tuple(int(c) for c in np.random.randint(20, 120, 3))
    cv2.rectangle(img, (6, 8), (58, 56), car_col, -1)
    # Windshield
    cv2.rectangle(img, (14, 12), (50, 24), (130, 140, 160), -1)
    # Rear window
    cv2.rectangle(img, (14, 40), (50, 52), (130, 140, 160), -1)
    # Wheels
    for wx, wy in [(8, 14), (50, 14), (8, 46), (50, 46)]:
        cv2.circle(img, (wx, wy), 5, (15, 15, 15), -1)
    # Lane markings
    cv2.line(img, (0, 0), (63, 0), (200, 200, 200), 2)
    cv2.line(img, (0, 63), (63, 63), (200, 200, 200), 2)
    # Slight brightness variation
    brightness = np.random.randint(-10, 10)
    img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────────
# 3. DEFAULT ROI POSITIONS
# ──────────────────────────────────────────────────────────────
def generate_roi_positions(img_w=1280, img_h=720) -> list:
    """Generate a grid of parking slot (x,y) positions."""
    positions = []
    for row_y in range(60, img_h - 60, SLOT_H + 12):
        for col in range((img_w - 100) // (SLOT_W + 10)):
            x = 50 + col * (SLOT_W + 10)
            y = row_y
            if x + SLOT_W < img_w and y + SLOT_H < img_h:
                positions.append((x, y))
    return positions


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def run():
    print("\n[START] Generating synthetic sample data ...")

    # Directories
    free_dir = RAW_DIR / "FREE"
    busy_dir = RAW_DIR / "BUSY"
    for d in [free_dir, busy_dir, PROC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Parking lot reference image
    lot_img = draw_parking_lot()
    lot_path = RAW_DIR / "parking_lot_frame.png"
    cv2.imwrite(str(lot_path), lot_img)
    print("  [OK] Parking lot image saved: " + str(lot_path))

    # 2. FREE slot images (200 samples)
    np.random.seed(0)
    for i in range(200):
        img = make_free_slot(i)
        cv2.imwrite(str(free_dir / f"free_{i:04d}.jpg"), img)
    print("  [OK] 200 FREE slot images saved -> " + str(free_dir))

    # 3. BUSY slot images (200 samples)
    np.random.seed(1)
    for i in range(200):
        img = make_busy_slot(i)
        cv2.imwrite(str(busy_dir / f"busy_{i:04d}.jpg"), img)
    print("  [OK] 200 BUSY slot images saved -> " + str(busy_dir))

    # 4. ROI positions pickle
    positions = generate_roi_positions()
    roi_path  = PROC_DIR / "parking_positions.pkl"
    with open(roi_path, "wb") as f:
        pickle.dump(positions, f)
    print("  [OK] " + str(len(positions)) + " ROI positions saved: " + str(roi_path))

    print("\n[DONE] Sample data ready.")
    print("  FREE  : " + str(len(list(free_dir.glob('*.jpg')))) + " images")
    print("  BUSY  : " + str(len(list(busy_dir.glob('*.jpg')))) + " images")
    print("  ROIs  : " + str(len(positions)) + " parking slots marked")
    print("\nNext step: python preprocess.py")


if __name__ == "__main__":
    run()
