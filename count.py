import os

TRAIN_DIR = "/Users/elaine/Documents/BDD100k_data/dataset_ultimate/images/train"

day = real = fake = other = 0

with os.scandir(TRAIN_DIR) as it:
    for entry in it:
        if entry.name.startswith("day_"):
            day += 1
        elif entry.name.startswith("real_"):
            real += 1
        elif entry.name.startswith("fake_"):
            fake += 1
        else:
            other += 1

print(f"Real Day   (day_):  {day}")
print(f"Real Night (real_): {real}")
print(f"Fake Night (fake_): {fake}")
if other:
    print(f"Khác:               {other}")
print(f"TỔNG:               {day + real + fake + other}")
