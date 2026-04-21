"""
===========================================================
 Smart Parking System - Real-Time Detection
 File   : realtime_detection.py
 Author : Smart Parking Team
 Purpose: Live parking detection from webcam or video file
          using RT-DETRv2. Displays bounding boxes, counts,
          and FPS in real time.
===========================================================

USAGE:
  # Webcam (default camera)
  python realtime_detection.py

  # Specific video file
  python realtime_detection.py --source ../data/raw/parking_lot.mp4

  # Webcam index (if multiple cameras)
  python realtime_detection.py --source 1

  Press 'q' to quit | 's' to save screenshot | 'p' to pause
===========================================================
"""

import cv2
import sys
import argparse
import time
import os
import numpy as np
from pathlib import Path
from collections import deque

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH   = Path("../models/best.pt")
OUTPUT_DIR   = Path("../outputs/predictions")
CONF_THRESH  = 0.30
IOU_THRESH   = 0.45
IMG_SIZE     = 640

# Display
WINDOW_NAME  = "Smart Parking — Real-Time AI Detection"
FPS_BUFFER   = 30     # frames to average FPS over

# Colours (BGR)
FREE_COLOR   = (0, 210, 80)
OCC_COLOR    = (0, 60, 220)
HUD_BG       = (20, 20, 20)
TEXT_COLOR   = (240, 240, 240)

CLASS_LABELS = {0: "FREE", 1: "OCCUPIED"}
CLASS_COLORS = {0: FREE_COLOR, 1: OCC_COLOR}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Smart Parking Detection"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0/1/2 = webcam, or path to video file"
    )
    parser.add_argument(
        "--model", default=str(MODEL_PATH),
        help="Path to trained model (.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESH,
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", default=False,
        help="Open in fullscreen mode"
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load the trained RT-DETRv2 model."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Model not found: {path}")
        print("  → Train first: python train_rtdetr.py")
        sys.exit(1)

    print(f"[INFO] Loading model: {path.name}")
    try:
        model = RTDETR(str(path))
    except Exception:
        model = YOLO(str(path))

    # Warmup
    print("[INFO] Warming up model ...")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[INFO] Model ready.")
    return model


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a video capture from webcam index or file path."""
    try:
        src = int(source)   # webcam index
    except ValueError:
        src = source        # file path

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    # Set webcam resolution
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Source opened: {w}×{h} @ {fps:.1f} FPS")
    return cap


def draw_hud(frame: np.ndarray, free: int, occupied: int,
             fps: float, paused: bool) -> None:
    """Draw HUD overlay: top banner with counts and FPS."""
    h, w = frame.shape[:2]
    total = free + occupied

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 70), HUD_BG, -1)

    # Title
    cv2.putText(frame, "SMART PARKING  |  RT-DETRv2  |  REAL-TIME",
                (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (160, 200, 255), 1)

    # Counts
    status_text = (f"FREE: {free}   OCCUPIED: {occupied}   "
                   f"TOTAL: {total}   FPS: {fps:.1f}"
                   + ("  [PAUSED]" if paused else ""))
    cv2.putText(frame, status_text, (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 1)

    # Availability progress bar
    if total > 0:
        bar_x  = w - 230
        bar_y  = 12
        bar_w  = 210
        bar_h  = 18
        filled = int(bar_w * free / total)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), FREE_COLOR, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        pct = f"{int(100 * free / total)}% available"
        cv2.putText(frame, pct,
                    (bar_x + 50, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Controls hint
    cv2.putText(frame, "[Q] Quit  [S] Screenshot  [P] Pause",
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def draw_boxes(frame: np.ndarray, result) -> tuple:
    """Draw detection boxes and return (free_count, occupied_count)."""
    free = 0
    occupied = 0

    if result.boxes is None or len(result.boxes) == 0:
        return free, occupied

    boxes     = result.boxes.xyxy.cpu().numpy()
    confs     = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label  = CLASS_LABELS.get(cls_id, "?")
        text   = f"{label} {conf:.2f}"

        if cls_id == 0:
            free += 1
        else:
            occupied += 1

        # Box
        cv2.rectangle(frame, (x1, max(y1, 71)), (x2, y2), color, 2)

        # Label background
        lbl_y = max(y1, 75)
        cv2.rectangle(frame,
                      (x1, lbl_y - 20),
                      (x1 + len(text) * 9 + 4, lbl_y - 2),
                      color, -1)

        # Label text
        cv2.putText(frame, text, (x1 + 2, lbl_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return free, occupied


def run() -> None:
    """Main real-time detection loop."""
    args  = parse_args()
    model = load_model(args.model)
    cap   = open_capture(args.source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # FPS tracking
    fps_buf    = deque(maxlen=FPS_BUFFER)
    prev_time  = time.time()

    # State
    paused     = False
    frame_num  = 0

    # Window setup
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    print(f"\n[INFO] Real-time detection started.")
    print("  Press Q to quit | S for screenshot | P to pause\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(args.source, str) and args.source != "0":
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[INFO] Stream ended.")
                    break

            frame_num += 1

            # ── INFERENCE ──────────────────────────────────────────────────────
            results = model.predict(
                source  = frame,
                conf    = args.conf,
                iou     = args.iou,
                imgsz   = IMG_SIZE,
                verbose = False,
            )
            result = results[0]

            # ── DRAW BOXES ─────────────────────────────────────────────────────
            free, occupied = draw_boxes(frame, result)

            # ── FPS ────────────────────────────────────────────────────────────
            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = np.mean(fps_buf)

            # ── HUD ────────────────────────────────────────────────────────────
            draw_hud(frame, free, occupied, fps, paused)

        # Display
        cv2.imshow(WINDOW_NAME, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] User quit.")
            break
        elif key == ord("p"):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = OUTPUT_DIR / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"[✓] Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_num} frames.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
