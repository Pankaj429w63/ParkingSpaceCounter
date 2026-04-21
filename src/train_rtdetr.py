"""
===========================================================
 Smart Parking System - RT-DETRv2 Training Script
 File   : train_rtdetr.py
 Author : Smart Parking Team
 Purpose: Fine-tune RT-DETRv2 on the parking dataset using
          Ultralytics framework with transfer learning.
          Includes NaN protection, stable LR schedule,
          and automatic best model saving.
===========================================================

USAGE:
  python train_rtdetr.py

Requires:
  - pip install ultralytics torch torchvision
  - Preprocessed dataset (run preprocess.py first)
  - data/processed/dataset.yaml
  - GPU strongly recommended (CUDA or Apple MPS)
===========================================================
"""

import os
import sys
import yaml
import torch
import shutil
from pathlib import Path
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET_YAML = Path("../data/processed/dataset.yaml")
MODEL_DIR    = Path("../models")
OUTPUT_DIR   = Path("../outputs/graphs")

# RT-DETRv2 model variant:
#   rtdetr-l.pt  → Large  (most accurate, needs >8 GB VRAM)
#   rtdetr-x.pt  → XLarge (best, needs >16 GB VRAM)
# For CPU/low-VRAM: use YOLOv8s as fallback (flag below)
MODEL_VARIANT  = "rtdetr-l.pt"
FALLBACK_MODEL = "yolov8s.pt"   # used when CUDA unavailable

# Training hyperparameters
EPOCHS       = 50        # Increase to 100 for better accuracy
IMG_SIZE     = 640
BATCH_SIZE   = 8         # Reduce to 4 if GPU OOM
LR0          = 0.0001    # Low LR for transfer learning stability
LRF          = 0.01      # Final LR = LR0 * LRF
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
PATIENCE     = 20        # Early stopping patience (epochs without improvement)
WORKERS      = 2         # DataLoader workers (0 on Windows if issues arise)
PROJECT_NAME = "SmartParking_RTDETRv2"

SAVE_PERIOD  = 5         # Save checkpoint every N epochs
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def check_environment() -> bool:
    """Check and print environment info before training."""
    print("\n" + "=" * 60)
    print("  ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM    : {vram_gb:.1f} GB")
    else:
        print("  GPU     : Not available → Training on CPU (slower)")
    print(f"  Device  : {DEVICE}")
    print("=" * 60 + "\n")
    return True


def resolve_dataset_yaml() -> str:
    """
    Ensure dataset.yaml exists and that paths within it are absolute.
    Returns the absolute path to the yaml file.
    """
    yaml_path = DATASET_YAML.resolve()
    if not yaml_path.exists():
        print(f"[ERROR] Dataset YAML not found: {yaml_path}")
        print("  → Run preprocess.py first to generate the dataset.")
        sys.exit(1)

    # Read and patch relative paths → absolute
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    base = yaml_path.parent.parent  # data/ folder
    config["path"] = str(base.resolve())

    # Write patched yaml to a temp location
    patched = yaml_path.parent / "_dataset_abs.yaml"
    with open(patched, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] Dataset YAML resolved: {patched}")
    return str(patched)


def select_model() -> str:
    """
    Select the best available model variant.
    Falls back to YOLOv8s if no GPU or low VRAM.
    """
    if DEVICE == "cpu":
        print(f"[WARN] No GPU detected. Using fallback model: {FALLBACK_MODEL}")
        print("  → Training will be slow. Consider Google Colab for GPU training.")
        return FALLBACK_MODEL

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 6:
            print(f"[WARN] Low VRAM ({vram_gb:.1f}GB). Using fallback: {FALLBACK_MODEL}")
            return FALLBACK_MODEL

    print(f"[INFO] Using model: {MODEL_VARIANT}")
    return MODEL_VARIANT


def copy_best_model(run_dir: Path) -> None:
    """Copy the best.pt from training run to models/ directory."""
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dst = MODEL_DIR / "best.pt"
        shutil.copy2(best_src, dst)
        print(f"\n[✓] Best model saved to: {dst}")
    else:
        print(f"[WARN] best.pt not found at: {best_src}")

    # Also copy last.pt as backup
    last_src = run_dir / "weights" / "last.pt"
    if last_src.exists():
        shutil.copy2(last_src, MODEL_DIR / "last.pt")
        print(f"[✓] Last checkpoint saved to: {MODEL_DIR / 'last.pt'}")


def run() -> None:
    """Main training pipeline."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    check_environment()

    dataset_yaml = resolve_dataset_yaml()
    model_name   = select_model()
    run_name     = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load pretrained model
    print(f"\n[INFO] Loading pretrained model: {model_name}")
    try:
        if "rtdetr" in model_name.lower():
            model = RTDETR(model_name)   # RT-DETRv2 via Ultralytics
        else:
            model = YOLO(model_name)    # Fallback YOLO
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"\n[INFO] Starting training ...")
    print(f"  Model     : {model_name}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Device    : {DEVICE}")
    print(f"  Dataset   : {dataset_yaml}")
    print("")

    # ─── TRAINING ──────────────────────────────────────────────────────────────
    results = model.train(
        data        = dataset_yaml,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        device      = DEVICE,
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay= WEIGHT_DECAY,
        warmup_epochs= WARMUP_EPOCHS,
        patience    = PATIENCE,
        workers     = WORKERS,
        project     = PROJECT_NAME,
        name        = run_name,
        save_period = SAVE_PERIOD,
        exist_ok    = True,
        verbose     = True,
        # Augmentation settings (mild to avoid overfitting on small datasets)
        augment     = True,
        fliplr      = 0.5,
        flipud      = 0.0,
        degrees     = 5.0,
        scale       = 0.3,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        # NaN prevention
        amp         = False if DEVICE == "cpu" else True,  # mixed precision
    )

    print("\n[INFO] Training complete!")

    # Locate run directory
    run_dir = Path(PROJECT_NAME) / run_name
    copy_best_model(run_dir)

    # Print final metrics
    try:
        metrics = results.results_dict
        print("\n" + "=" * 50)
        print("  TRAINING SUMMARY")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:<35}: {v:.4f}")
        print("=" * 50)
    except Exception:
        pass  # Metrics not always available for all model types

    print("\n[DONE] Model ready at: ../models/best.pt")
    print("  Next step: python evaluate.py")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
