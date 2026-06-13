"""
Calculate FID between:
- Fake Night images (trainA_night) in dataset_night/images/train
- Real Night images in dataset_night/images/val
"""

import os, json, shutil
from pathlib import Path
from cleanfid import fid

# ── Configuration ─────────────────────────────────────────────────────────────
BASE    = Path("/workspace/dataset_night")
VAL_JSON = Path("/workspace/BDD100k_data/labels/bdd100k_labels_images_val.json")

# Temp folders for clean-fid
DIR_REAL = Path("/tmp/fid_real")
DIR_FAKE = Path("/tmp/fid_fake")

# ── Cleanup & Setup ───────────────────────────────────────────────────────────
for d in [DIR_REAL, DIR_FAKE]:
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)

# ── 1. Identify Real Night images from Val ────────────────────────────────────
print("Filtering Real Night images from Val set...")
with open(VAL_JSON) as f:
    val_data = json.load(f)

# Keep only 'night' and 'dawn/dusk' from validation set
real_night_names = set()
for item in val_data:
    tod = item.get("attributes", {}).get("timeofday", "")
    if tod in ("night", "dawn/dusk"):
        real_night_names.add(item["name"])

ds_val_img = BASE / "images" / "val"
count_real = 0
for img_path in ds_val_img.glob("*.jpg"):
    if img_path.name in real_night_names:
        os.symlink(img_path, DIR_REAL / img_path.name)
        count_real += 1

print(f"  Found {count_real} real night images in val.")

# ── 2. Identify Fake Night images from Train ──────────────────────────────────
# We know trainB are real night, so everything else in images/train is fake night
print("Filtering Fake Night images from Train set...")
TRAINB_DIR = Path("/workspace/BDD100k_data/images/train/trainB")
trainb_names = set(f.name for f in TRAINB_DIR.glob("*.jpg"))

ds_train_img = BASE / "images" / "train"
count_fake = 0
for img_path in ds_train_img.glob("*.jpg"):
    if img_path.name not in trainb_names:
        os.symlink(img_path, DIR_FAKE / img_path.name)
        count_fake += 1

print(f"  Found {count_fake} fake night images in train.")

# ── 3. Calculate FID ──────────────────────────────────────────────────────────
if count_real > 0 and count_fake > 0:
    print(f"\n🚀 Calculating FID (clean-fid)...")
    score = fid.compute_fid(str(DIR_REAL), str(DIR_FAKE))
    print(f"\n✅ FID Score: {score:.4f}")
else:
    print("❌ Error: Missing images for calculation.")

# ── Cleanup ───────────────────────────────────────────────────────────────────
shutil.rmtree(DIR_REAL)
shutil.rmtree(DIR_FAKE)
