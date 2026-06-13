import os
from pathlib import Path

BASE = Path("/workspace/")
DATASETS = ["dataset_baseline"]

def cleanup(dataset_name):
    print(f"\nCleaning {dataset_name}...")
    root = BASE / dataset_name
    count = 0
    
    # Duyệt qua cả train và val labels
    for split in ["train", "val"]:
        label_dir = root / "labels" / split
        image_dir = root / "images" / split
        
        if not label_dir.exists(): continue
        
        for label_path in label_dir.glob("*.txt"):
            # Kiểm tra nếu file trống
            if label_path.stat().st_size == 0:
                img_path = image_dir / (label_path.stem + ".jpg")
                
                # Xóa cả nhãn và ảnh
                if label_path.exists():
                    label_path.unlink()
                if img_path.exists():
                    img_path.unlink()
                
                count += 1
    
    print(f"  Removed {count} images and labels because they were empty.")

for ds in DATASETS:
    cleanup(ds)
