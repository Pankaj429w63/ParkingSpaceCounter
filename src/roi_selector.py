"""
===========================================================
 Smart Parking System - ROI Selector
 File   : roi_selector.py
 Author : Smart Parking Team
 Purpose: Allow user to mark parking slot positions on a
          reference frame using mouse clicks. Positions are
          saved to a pickle file for use in opencv_parking.py
===========================================================

HOW TO USE:
  python roi_selector.py

  LEFT CLICK  → Add a new parking slot at cursor position
  RIGHT CLICK → Remove the nearest parking slot
  Press 's'   → Save positions and exit
  Press 'r'   → Reset (delete all positions)
  Press 'q'   → Quit without saving
"""

import cv2
import pickle
import os
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
VIDEO_PATH   = "../data/raw/parking_lot.mp4"
IMAGE_PATH   = "../data/raw/parking_lot_frame.png"  # static reference image
ROI_FILE     = "../data/processed/parking_positions.pkl"
SLOT_WIDTH   = 107
SLOT_HEIGHT  = 48

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
positions: list = []   # list of (x, y) tuples
drawing_mode = False   # future: drag-to-draw mode


def load_existing(roi_file: str) -> list:
    """Load previously saved positions if file exists."""
    if os.path.exists(roi_file):
        with open(roi_file, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] Loaded {len(data)} existing positions from {roi_file}")
        return data
    return []


def save_positions(roi_file: str, pos_list: list) -> None:
    """Save current positions to pickle file."""
    os.makedirs(os.path.dirname(roi_file), exist_ok=True)
    with open(roi_file, "wb") as f:
        pickle.dump(pos_list, f)
    print(f"[INFO] Saved {len(pos_list)} positions to {roi_file}")


def get_nearest_index(pos_list: list, mx: int, my: int,
                      width: int, height: int) -> int:
    """
    Find index of the slot whose bounding box contains (mx, my).
    Returns -1 if none found.
    """
    for i, (x, y) in enumerate(pos_list):
        if x < mx < x + width and y < my < y + height:
            return i
    return -1


def mouse_callback(event, mx, my, flags, param):
    """Handle mouse click events for adding/removing slots."""
    global positions

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add new slot (top-left = click position)
        positions.append((mx, my))
        print(f"  [+] Added slot at ({mx}, {my})  |  Total: {len(positions)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove slot whose bounding box contains the click
        idx = get_nearest_index(positions, mx, my, SLOT_WIDTH, SLOT_HEIGHT)
        if idx != -1:
            removed = positions.pop(idx)
            print(f"  [-] Removed slot at {removed}  |  Total: {len(positions)}")


def draw_slots(img: np.ndarray) -> np.ndarray:
    """Draw all current slots on a copy of the image."""
    display = img.copy()

    for i, (x, y) in enumerate(positions):
        # Alternating tint for easier visibility
        color = (0, 200, 80) if i % 2 == 0 else (0, 150, 255)
        cv2.rectangle(display, (x, y),
                      (x + SLOT_WIDTH, y + SLOT_HEIGHT), color, 2)
        cv2.putText(display, str(i + 1), (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Info banner
    cv2.rectangle(display, (0, 0), (display.shape[1], 55), (30, 30, 30), -1)
    cv2.putText(display, "ROI SELECTOR — Left Click: Add | Right Click: Remove",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(display,
                f"Slots: {len(positions)}   [S] Save & Exit   [R] Reset   [Q] Quit",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1)

    return display


def get_reference_image() -> np.ndarray:
    """
    Get a reference frame either from a saved image or the first video frame.
    Priority: static image → first video frame → blank canvas.
    """
    if os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH)
        if img is not None:
            print(f"[INFO] Using static reference image: {IMAGE_PATH}")
            return img

    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"[INFO] Captured reference frame from video: {VIDEO_PATH}")
            cv2.imwrite(IMAGE_PATH, frame)   # Save for future use
            return frame

    print("[WARNING] No video or image found. Using blank canvas (600×800).")
    return np.zeros((600, 800, 3), dtype=np.uint8)


def run() -> None:
    """Main ROI selection loop."""
    global positions

    ref_img = get_reference_image()
    positions = load_existing(ROI_FILE)

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    print("\n[CONTROLS]")
    print("  Left Click  → Add slot")
    print("  Right Click → Remove slot")
    print("  S           → Save & exit")
    print("  R           → Reset all slots")
    print("  Q           → Quit without saving\n")

    while True:
        display = draw_slots(ref_img)
        cv2.imshow("ROI Selector", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            save_positions(ROI_FILE, positions)
            print("[INFO] Positions saved. Exiting selector.")
            break
        elif key == ord("r"):
            positions = []
            print("[INFO] All positions cleared.")
        elif key == ord("q"):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyAllWindows()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
