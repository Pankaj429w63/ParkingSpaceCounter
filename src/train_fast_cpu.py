"""
===========================================================
 Smart Parking - Fast CPU Training (Quick Demo)
 File   : train_fast_cpu.py
 Purpose: Train YOLOv8n (Nano) for 5 epochs on CPU.
          Completes in ~10-20 minutes (vs hours for RT-DETR).
          Produces a working best.pt for the dashboard demo.

 For FULL accuracy (93-98%), use:
   python src/train_rtdetr.py   (needs GPU)
   OR upload to Google Colab for free GPU

 This script is for LOCAL DEMO purposes only.
===========================================================
"""

import sys
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent
YAML  = _BASE / "data" / "processed" / "dataset.yaml"
MODEL_DIR = _BASE / "models"

EPOCHS     = 5      # Increase to 50+ for real accuracy
IMG_SIZE   = 320    # Smaller for CPU speed (640 for full accuracy)
BATCH_SIZE = 4
MODEL_NAME = "yolov8n.pt"   # Nano = smallest & fastest
PROJECT    = str(_BASE / "runs" / "train")
RUN_NAME   = "parking_cpu_demo"


def run():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    if not YAML.exists():
        print("[ERROR] dataset.yaml not found. Run preprocess.py first.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  SMART PARKING — CPU Fast Training (YOLOv8n)")
    print("=" * 55)
    print("  Model     : YOLOv8n (Nano — optimised for CPU)")
    print("  Epochs    : " + str(EPOCHS) + "  (use 50+ for production)")
    print("  Image Size: " + str(IMG_SIZE) + "  (use 640 for production)")
    print("  Device    : cpu")
    print("=" * 55)
    print("\n  Starting training (this may take 10-20 minutes) ...\n")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data        = str(YAML),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        device      = "cpu",
        workers     = 0,       # 0 = main thread only (safe on Windows)
        project     = PROJECT,
        name        = RUN_NAME,
        exist_ok    = True,
        verbose     = True,
        amp         = False,   # No mixed precision on CPU
        # Light augmentation
        fliplr      = 0.5,
        degrees     = 5.0,
        scale       = 0.2,
        patience    = 10,
        save_period = 1,
    )

    # Copy best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_src = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODEL_DIR / "best.pt")
        print("\n[OK] Best model saved -> models/best.pt")
    else:
        last_src = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(last_src, MODEL_DIR / "best.pt")
            print("\n[OK] Last model saved -> models/best.pt")

    print("\n[DONE] Training complete!")
    print("  model -> models/best.pt")
    print("  Next  -> streamlit run src/dashboard.py")


if __name__ == "__main__":
    run()
