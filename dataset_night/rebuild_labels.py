import json
import os
from pathlib import Path

BASE = Path("/Users/elaine/Documents/BDD100k_data")
OUT_DIR = BASE / "dataset_night"

# Map exactly to Baseline order
CATEGORY_MAP = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "bus": 3,
    "truck": 4,
    "bike": 5,
    "motor": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9,
}

IMG_W, IMG_H = 1280, 720

def box2yolo(x1, y1, x2, y2):
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w  = (x2 - x1) / IMG_W
    h  = (y2 - y1) / IMG_H
    return cx, cy, w, h

def rebuild_labels(json_file, image_folder, output_label_folder):
    print(f"Loading {json_file.name}...")
    with open(json_file) as f:
        data = json.load(f)
    
    label_lookup = {}
    for item in data:
        name = item["name"]
        lines = []
        for frame_labels in item.get("labels", []) or []:
            cat = frame_labels.get("category", "")
            if cat not in CATEGORY_MAP:
                continue
            box2d = frame_labels.get("box2d")
            if not box2d:
                continue
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            cx, cy, w, h = box2yolo(x1, y1, x2, y2)
            # Clamp
            cx = min(max(cx, 0), 1)
            cy = min(max(cy, 0), 1)
            w  = min(max(w,  0), 1)
            h  = min(max(h,  0), 1)
            if w > 0 and h > 0:
                lines.append(f"{CATEGORY_MAP[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_lookup[name] = lines

    print(f"Processing labels for images in {image_folder}...")
    output_label_folder.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in sorted(image_folder.glob("*.jpg")):
        lines = label_lookup.get(img_path.name)
        if lines is not None:
            dst_lbl = output_label_folder / (img_path.stem + ".txt")
            dst_lbl.write_text("\n".join(lines))
            count += 1
    print(f"Done. Rebuilt {count} labels in {output_label_folder}")

# Main execution
if __name__ == "__main__":
    # 1. VAL
    rebuild_labels(
        BASE / "labels" / "bdd100k_labels_images_val.json",
        OUT_DIR / "images" / "val",
        OUT_DIR / "labels" / "val"
    )
    
    # 2. TRAIN
    rebuild_labels(
        BASE / "labels" / "bdd100k_labels_images_train.json",
        OUT_DIR / "images" / "train",
        OUT_DIR / "labels" / "train"
    )
