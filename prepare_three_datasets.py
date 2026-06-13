import json
import os
import shutil
from pathlib import Path
import random

BASE = Path("/Users/elaine/Documents/BDD100k_data")
LABEL_VAL_JSON = BASE / "labels" / "bdd100k_labels_images_val.json"
LABEL_TRAIN_JSON = BASE / "labels" / "bdd100k_labels_images_train.json"

# Chuẩn 10 lớp BDD100k
CATEGORY_MAP = {
    "person": 0, "rider": 1, "car": 2, "bus": 3, "truck": 4,
    "bike": 5, "motor": 6, "traffic light": 7, "traffic sign": 8, "train": 9
}

IMG_W, IMG_H = 1280, 720

def box2yolo(x1, y1, x2, y2):
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w  = (x2 - x1) / IMG_W
    h  = (y2 - y1) / IMG_H
    return cx, cy, w, h

def load_labels(json_path):
    print(f"Loading {json_path.name}...")
    with open(json_path) as f:
        data = json.load(f)
    lookup = {}
    for item in data:
        lines = []
        for label in item.get("labels", []) or []:
            cat = label.get("category", "")
            if cat in CATEGORY_MAP and label.get("box2d"):
                b = label["box2d"]
                cx, cy, w, h = box2yolo(b["x1"], b["y1"], b["x2"], b["y2"])
                lines.append(f"{CATEGORY_MAP[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lookup[item["name"]] = lines
    return lookup

def create_yaml(path, nc, names):
    content = f"path: {path.absolute()}\ntrain: images/train\nval: images/val\n\nnc: {nc}\nnames:\n"
    for i, name in enumerate(names):
        content += f"  {i}: {name}\n"
    (path / "dataset.yaml").write_text(content)

def build_subset(name, train_images, val_images, train_labels, val_labels):
    print(f"\nBuilding {name}...")
    root = BASE / name
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    
    # Copy/Link Train
    for img_path in train_images:
        dst_img = root / "images/train" / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        lines = train_labels.get(img_path.name, [])
        (root / "labels/train" / (img_path.stem + ".txt")).write_text("\n".join(lines))
    
    # Copy/Link Val
    for img_path in val_images:
        dst_img = root / "images/val" / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        lines = val_labels.get(img_path.name, [])
        (root / "labels/val" / (img_path.stem + ".txt")).write_text("\n".join(lines))
    
    create_yaml(root, 10, list(CATEGORY_MAP.keys()))
    print(f"Done {name}")

# Main
train_labels = load_labels(LABEL_TRAIN_JSON)
val_labels = load_labels(LABEL_VAL_JSON)

trainA = sorted(list((BASE / "images/train/trainA").glob("*.jpg")))
trainB = sorted(list((BASE / "images/train/trainB").glob("*.jpg")))
val_night = sorted(list((BASE / "images/val/valB").glob("*.jpg"))) 
# Note: Using standard val set images if valB exists, or just use the night ones from val.
# Let's check where the val images are.
val_dir = BASE / "dataset_night" / "images" / "val"
val_images = sorted(list(val_dir.glob("*.jpg")))

# 1. dataset_day_yolo
build_subset("dataset_day_yolo", trainA, val_images, train_labels, val_labels)

# 2. dataset_night_yolo
build_subset("dataset_night_yolo", trainB, val_images, train_labels, val_labels)

# 3. dataset_baseline
# Balanced: 24750 Night + 24750 Day
random.seed(42)
day_subset = random.sample(trainA, 24750)
baseline_train = trainB + day_subset
build_subset("dataset_baseline", baseline_train, val_images, train_labels, val_labels)
