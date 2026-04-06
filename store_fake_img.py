import os
import shutil
import glob

# Đường dẫn kết quả vừa test xong (Elaine check lại tên folder nhé)
source_dir = './results/bdd_final_test/bdd_resnet9_10ep/test_latest/images'
target_dir = './results/fake_B_only'

os.makedirs(target_dir, exist_ok=True)

# Lấy tất cả file có đuôi _fake_B.png
fake_files = glob.glob(os.path.join(source_dir, '*_fake_B.png'))

for f in fake_files:
    shutil.copy(f, os.path.join(target_dir, os.path.basename(f)))

print(f"Đã lọc xong {len(fake_files)} ảnh Fake_B vào thư mục {target_dir}")