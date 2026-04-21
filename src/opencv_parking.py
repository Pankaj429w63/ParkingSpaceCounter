"""
===========================================================
 Smart Parking System - OpenCV Baseline
 File   : opencv_parking.py
 Author : Smart Parking Team
 Purpose: Detect parking slots using traditional CV pipeline
          Video → Grayscale → Blur → Threshold → ROI → Pixel Count → Display
===========================================================
"""

import cv2
import pickle
import os
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
VIDEO_PATH   = "../data/raw/parking_lot.mp4"  # Path to input video
ROI_FILE     = "../data/processed/parking_positions.pkl"  # Saved ROI positions
SLOT_WIDTH   = 107   # Width of each parking slot (pixels)
SLOT_HEIGHT  = 48    # Height of each parking slot (pixels)

# Threshold: if non-zero pixels in ROI > THRESH → slot is OCCUPIED
PIXEL_THRESHOLD = 900

# Colours (BGR)
FREE_COLOR     = (0, 200, 0)    # Green
OCCUPIED_COLOR = (0, 0, 200)    # Red
TEXT_COLOR     = (255, 255, 255) # White
OVERLAY_COLOR  = (50, 50, 50)    # Dark overlay


def load_positions(roi_file: str) -> list:
    """Load saved parking slot positions from pickle file."""
    if not os.path.exists(roi_file):
        print(f"[WARNING] ROI file not found: {roi_file}")
        print(" → Run roi_selector.py first to mark parking slots.")
        return []
    with open(roi_file, "rb") as f:
        positions = pickle.load(f)
    print(f"[INFO] Loaded {len(positions)} parking slot positions.")
    return positions


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing pipeline to a video frame.
    Pipeline: BGR → Grayscale → GaussianBlur → AdaptiveThreshold
    Returns a binary (black/white) image for pixel counting.
    """
    # Step 1: Convert to grayscale (removes colour info, keeps intensity)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur to reduce noise (kernel 3x3, sigma=1)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)

    # Step 3: Adaptive Threshold → binary image
    #   - ADAPTIVE_THRESH_GAUSSIAN_C: uses weighted sum of neighbourhood
    #   - THRESH_BINARY_INV: white pixels = foreground (cars)
    #   - blockSize=25: neighbourhood size; C=16: constant subtracted
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 16
    )

    # Step 4: Median blur to clean up small noise speckles
    thresh = cv2.medianBlur(thresh, 5)

    # Step 5: Dilation to connect broken edges of cars
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh


def classify_slot(thresh_frame: np.ndarray, pos: tuple,
                  width: int, height: int) -> tuple:
    """
    Classify a single parking slot as FREE or OCCUPIED.

    Args:
        thresh_frame : Preprocessed binary frame
        pos          : (x, y) top-left corner of slot
        width, height: Slot dimensions

    Returns:
        (is_free: bool, pixel_count: int)
    """
    x, y = pos

    # Extract the Region of Interest (ROI) for this slot
    roi = thresh_frame[y : y + height, x : x + width]

    # Count white (foreground) pixels — more white = more car present
    pixel_count = cv2.countNonZero(roi)

    # Classify: if pixel count exceeds threshold → OCCUPIED
    is_free = pixel_count < PIXEL_THRESHOLD

    return is_free, pixel_count


def draw_slot(frame: np.ndarray, pos: tuple, is_free: bool,
              pixel_count: int, width: int, height: int) -> None:
    """Draw bounding box and label on the original frame for one slot."""
    x, y = pos
    color = FREE_COLOR if is_free else OCCUPIED_COLOR
    label = "FREE" if is_free else "BUSY"
    thickness = 2 if is_free else 2

    # Draw rectangle around slot
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)

    # Draw semi-transparent fill for better visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Label text inside the slot
    cv2.putText(
        frame, label,
        (x + 2, y + height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        TEXT_COLOR, 1, cv2.LINE_AA
    )

    # Pixel count (debug mode)
    # cv2.putText(frame, str(pixel_count), (x+2, y+20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)


def draw_summary_bar(frame: np.ndarray, free: int, total: int) -> None:
    """Draw a top status bar showing parking availability summary."""
    h, w = frame.shape[:2]
    occupied = total - free

    # Dark banner at top
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    # Title
    cv2.putText(frame, "SMART PARKING SYSTEM",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

    # Counts
    summary = f"FREE: {free}   OCCUPIED: {occupied}   TOTAL: {total}"
    cv2.putText(frame, summary, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Availability bar (green portion = free)
    if total > 0:
        bar_x, bar_y, bar_h = w - 220, 10, 20
        bar_w = 200
        filled = int(bar_w * free / total)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      FREE_COLOR, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 1)
        pct_text = f"{int(100 * free / total)}% free"
        cv2.putText(frame, pct_text, (bar_x + 60, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def run(video_path: str = VIDEO_PATH, roi_file: str = ROI_FILE) -> None:
    """
    Main loop: open video, process each frame, display results.
    Press 'q' to quit.
    """
    # Load slot positions
    positions = load_positions(roi_file)
    if not positions:
        print("[ERROR] No parking positions available. Exiting.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_slots = len(positions)
    frame_count = 0
    print(f"[INFO] Processing video with {total_slots} slots. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # Preprocess (every frame, ~25 FPS capable)
        thresh = process_frame(frame)

        # Classify each slot
        free_count = 0
        for pos in positions:
            is_free, pixel_count = classify_slot(
                thresh, pos, SLOT_WIDTH, SLOT_HEIGHT)
            draw_slot(frame, pos, is_free, pixel_count,
                      SLOT_WIDTH, SLOT_HEIGHT)
            if is_free:
                free_count += 1

        # Draw summary banner
        draw_summary_bar(frame, free_count, total_slots)

        # Show frame
        cv2.imshow("Smart Parking System - OpenCV", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User exited.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames total.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
