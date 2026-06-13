import json
import os
import shutil
from pathlib import Path
import random

BASE = Path("/Users/elaine/Documents/BDD100k_data")
LABEL_TRAIN_JSON = BASE / "labels" / "bdd100k_labels_images_train.json"
LABEL_VAL_JSON = BASE / "labels" / "bdd100k_labels_images_val.json"
OUT_DIR = BASE / "dataset_ultimate"

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

# Prepare folders
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

train_labels = load_labels(LABEL_TRAIN_JSON)
val_labels = load_labels(LABEL_VAL_JSON)

# Source images
trainA = sorted(list((BASE / "images/train/trainA").glob("*.jpg")))
trainB = sorted(list((BASE / "images/train/trainB").glob("*.jpg")))
fakeN  = sorted(list((BASE / "outputs" / "trainA_night").glob("*.jpg")))
val_images = sorted(list((BASE / "dataset_night" / "images" / "val").glob("*.jpg")))

random.seed(42)
day_subset = random.sample(trainA, 24750)
fake_subset = random.sample(fakeN, 24750)
real_night = trainB # All 24750

def process_batch(images, prefix, lookup, split="train"):
    count = 0
    for img_path in images:
        lines = lookup.get(img_path.name, [])
        if not lines: continue # Skip empty if we want extreme clean
        
        # Copy image with prefix
        new_name = f"{prefix}_{img_path.name}"
        shutil.copy2(img_path, OUT_DIR / "images" / split / new_name)
        
        # Write label with prefix
        label_name = f"{prefix}_{img_path.stem}.txt"
        (OUT_DIR / "labels" / split / label_name).write_text("\n".join(lines))
        count += 1
    return count

print("\nBuilding dataset_ultimate...")
c1 = process_batch(day_subset, "day", train_labels)
print(f"  Added {c1} Real Day images.")

c2 = process_batch(real_night, "real", train_labels)
print(f"  Added {c2} Real Night images.")

c3 = process_batch(fake_subset, "fake", train_labels)
print(f"  Added {c3} Fake Night images.")

# Copy Val set (already prefixing if needed, but here simple is ok)
process_batch(val_images, "val", val_labels, split="val")

# Create YAML
content = f"path: {OUT_DIR.absolute()}\ntrain: images/train\nval: images/val\n\nnc: 10\nnames:\n"
for i, name in enumerate(CATEGORY_MAP.keys()):
    content += f"  {i}: {name}\n"
(OUT_DIR / "dataset.yaml").write_text(content)

print(f"\nDone! Ultimate dataset created at {OUT_DIR}")
