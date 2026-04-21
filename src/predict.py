"""
===========================================================
 Smart Parking System - Prediction Module
 File   : predict.py
 Author : Smart Parking Team
 Purpose: Run inference on single images or full folders.
          Draws bounding boxes, FREE/OCCUPIED labels, and
          saves annotated outputs to outputs/predictions/.
===========================================================

USAGE:
  # Single image
  python predict.py --source ../test.jpg

  # Folder of images
  python predict.py --source ../data/test/images/

  # With custom confidence threshold
  python predict.py --source ../test.jpg --conf 0.40
===========================================================
"""

import cv2
import sys
import argparse
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH  = Path("../models/best.pt")
OUTPUT_DIR  = Path("../outputs/predictions")
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 640

# Class display config
CLASS_COLORS = {
    0: (0, 200, 80),    # FREE  → Green (BGR)
    1: (0, 50, 220),    # OCCUPIED → Red (BGR)
}
CLASS_LABELS = {0: "FREE", 1: "OCCUPIED"}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Parking Prediction — RT-DETRv2"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to image file or folder of images"
    )
    parser.add_argument(
        "--model", default=str(MODEL_PATH),
        help="Path to trained model weights (.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help="Detection confidence threshold (0-1)"
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESH,
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save annotated output images"
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Display result windows (requires display)"
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load the trained Ultralytics model."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        print("[ERROR] Install ultralytics: pip install ultralytics")
        sys.exit(1)

    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Model not found: {path}")
        print("  → Run: python train_rtdetr.py")
        sys.exit(1)

    print(f"[INFO] Loading model: {path.name}")
    try:
        model = RTDETR(str(path))
    except Exception:
        model = YOLO(str(path))

    print("[INFO] Model loaded successfully.")
    return model


def collect_images(source: str) -> list:
    """
    Collect image paths from a file or directory.
    Returns sorted list of Path objects.
    """
    src = Path(source)
    if not src.exists():
        print(f"[ERROR] Source not found: {src}")
        sys.exit(1)

    if src.is_file():
        if src.suffix.lower() in VALID_EXTS:
            return [src]
        else:
            print(f"[ERROR] Unsupported file type: {src.suffix}")
            sys.exit(1)

    # Directory mode
    images = sorted([
        p for p in src.rglob("*")
        if p.suffix.lower() in VALID_EXTS
    ])
    print(f"[INFO] Found {len(images)} images in: {src}")
    return images


def draw_detections(img: np.ndarray, boxes, confs, class_ids) -> tuple:
    """
    Annotate image with bounding boxes and labels.

    Args:
        img       : Original BGR image
        boxes     : List of [x1, y1, x2, y2] boxes
        confs     : List of confidence scores
        class_ids : List of class indices

    Returns:
        (annotated_img, free_count, occupied_count)
    """
    annotated = img.copy()
    free_count     = 0
    occupied_count = 0

    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color   = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label   = CLASS_LABELS.get(cls_id, "UNKNOWN")
        display = f"{label} {conf:.2f}"

        if cls_id == 0:
            free_count += 1
        else:
            occupied_count += 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw semi-transparent filled rectangle for label background
        label_bg_y2 = y1 - 4
        label_bg_y1 = y1 - 22
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, label_bg_y1),
                      (x1 + len(display) * 9, label_bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Label text
        cv2.putText(annotated, display,
                    (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, free_count, occupied_count


def draw_summary(img: np.ndarray, free: int, occupied: int,
                 filename: str) -> np.ndarray:
    """Add a top summary banner to the annotated image."""
    h, w = img.shape[:2]
    total = free + occupied

    # Dark top banner
    cv2.rectangle(img, (0, 0), (w, 65), (25, 25, 25), -1)

    # Title
    cv2.putText(img, "SMART PARKING SYSTEM — AI Detection",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1)

    # Counts
    summary = (f"FREE: {free}   OCCUPIED: {occupied}   "
               f"TOTAL: {total}   File: {filename}")
    cv2.putText(img, summary, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


def predict_image(model, img_path: Path, conf: float,
                  iou: float) -> dict:
    """
    Run inference on a single image and return results dict.
    """
    # Run prediction
    results = model.predict(
        source    = str(img_path),
        conf      = conf,
        iou       = iou,
        imgsz     = IMG_SIZE,
        verbose   = False,
        save      = False,   # we handle saving manually
    )

    result = results[0]
    img = cv2.imread(str(img_path))

    # Extract detections
    boxes     = []
    confs     = []
    class_ids = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes     = result.boxes.xyxy.cpu().numpy().tolist()
        confs     = result.boxes.conf.cpu().numpy().tolist()
        class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

    # Annotate
    annotated, free, occupied = draw_detections(img, boxes, confs, class_ids)
    annotated = draw_summary(annotated, free, occupied, img_path.name)

    return {
        "image"    : annotated,
        "free"     : free,
        "occupied" : occupied,
        "total"    : free + occupied,
        "boxes"    : boxes,
        "confs"    : confs,
        "class_ids": class_ids,
    }


def save_output(annotated: np.ndarray, src_path: Path,
                out_dir: Path) -> Path:
    """Save annotated image to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{src_path.stem}{src_path.suffix}"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def run() -> None:
    """Main prediction pipeline."""
    args   = parse_args()
    model  = load_model(args.model)
    images = collect_images(args.source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_free     = 0
    total_occupied = 0
    processed      = 0

    print(f"\n[INFO] Running predictions on {len(images)} image(s) ...\n")

    for img_path in images:
        print(f"  Processing: {img_path.name}", end="  ")
        try:
            result = predict_image(model, img_path, args.conf, args.iou)

            total_free     += result["free"]
            total_occupied += result["occupied"]
            processed      += 1

            print(f"FREE={result['free']}  OCCUPIED={result['occupied']}", end="")

            if args.save:
                out_path = save_output(result["image"], img_path, OUTPUT_DIR)
                print(f"  → Saved: {out_path.name}", end="")

            if args.show:
                cv2.imshow(f"Prediction: {img_path.name}", result["image"])
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] User quit early.")
                    break

            print()   # newline

        except Exception as e:
            print(f"\n  [ERROR] Failed on {img_path.name}: {e}")
            continue

    if args.show:
        cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 55)
    print("  PREDICTION SUMMARY")
    print("=" * 55)
    print(f"  Images processed : {processed}/{len(images)}")
    print(f"  Total FREE slots : {total_free}")
    print(f"  Total OCCUPIED   : {total_occupied}")
    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print("=" * 55)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
