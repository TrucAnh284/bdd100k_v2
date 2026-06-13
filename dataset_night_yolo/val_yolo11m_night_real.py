"""
Validate YOLO11m on the dataset_night validation set to calculate mAP.
"""

from ultralytics import YOLO
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_CFG = BASE / "dataset.yaml"
# Đường dẫn tới file weights tốt nhất sau khi bạn train xong
WEIGHTS  = BASE / "runs" / "yolo11m_night_real_40ep" / "weights" / "best.pt"

# ── Load Model ────────────────────────────────────────────────────────────────
if not WEIGHTS.exists():
    print(f"❌ Error: Không tìm thấy file weights tại {WEIGHTS}")
    print("Mẹo: Hãy đảm bảo bạn đã train xong hoặc điều chỉnh đường dẫn WEIGHTS cho đúng.")
else:
    model = YOLO(str(WEIGHTS))

    # ── Run Validation ────────────────────────────────────────────────────────
    print(f"🚀 Đang đánh giá model trên tập validation: {DATA_CFG}")
    results = model.val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=16,
        conf=0.001,  # Confidence threshold thấp để tính mAP chính xác
        iou=0.6,     # NMS IoU threshold
        device=0,    # GPU 0
        split='val', # Sử dụng tập val định nghĩa trong dataset.yaml
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8] # Loại bỏ class "train" (ID 5) do quá ít dữ liệu
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n=== Kết quả đánh giá mAP ===")
    print(f"mAP@50:      {results.results_dict['metrics/m_ap50']:.4f}")
    print(f"mAP@50-95:   {results.results_dict['metrics/m_ap']:.4f}")
    print(f"Chi tiết lưu tại: {results.save_dir}")
