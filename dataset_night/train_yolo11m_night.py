"""
Train YOLO11m on dataset_night (day→night augmented dataset)
Server: RTX 4090 (24GB VRAM)
"""

from ultralytics import YOLO
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent          # dataset_night/
DATA_CFG = ROOT / "dataset.yaml"
PROJECT  = ROOT / "runs"
NAME     = "yolo11m_night_40ep"

# ── Model ─────────────────────────────────────────────────────────────────────
model = YOLO("yolo11m.pt")               # downloads pretrained weights if needed

# ── Training ──────────────────────────────────────────────────────────────────
model.train(
    data        = str(DATA_CFG),
    epochs      = 40,
    imgsz       = 640,                  # BDD100k native resolution
    batch       = 16,                     # RTX 4090 24GB: safe at imgsz=1280
    workers     = 8,
    device      = 0,                     # GPU 0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer   = "AdamW",
    lr0         = 1e-3,
    lrf         = 0.01,                  # final lr = lr0 * lrf
    momentum    = 0.937,
    weight_decay= 5e-4,
    warmup_epochs     = 3,
    warmup_momentum   = 0.8,
    warmup_bias_lr    = 0.1,

    # ── Augmentation ──────────────────────────────────────────────────────────
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 0.0,
    translate   = 0.1,
    scale       = 0.5,
    flipud      = 0.0,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.1,

    # ── Logging / Saving ──────────────────────────────────────────────────────
    project     = str(PROJECT),
    name        = NAME,
    exist_ok    = False,
    save        = True,
    save_period = 5,                     # checkpoint every 5 epochs
    val         = True,
    plots       = True,
    verbose     = True,
)

print(f"\n✅ Training complete. Weights saved to: {PROJECT / NAME / 'weights'}")
