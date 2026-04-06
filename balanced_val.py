import os
import json
import random

# CẤU HÌNH ĐƯỜNG DẪN CHUẨN THEO ẢNH CỦA ELAINE
ROOT_DATA = '/workspace/BDD100k_data' # Folder chứa labels
VAL_IMG_DIR = '/workspace/BDD100k_data/images/val' # Folder chứa ảnh val thực tế
OUTPUT_PATH = '/workspace/datasets/bdd_final'

def run_balance():
    label_path = os.path.join(ROOT_DATA, 'labels/bdd100k_labels_images_val.json')
    
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file nhãn tại: {label_path}")
        return

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # 1. Phân loại và kiểm tra sự tồn tại của file ảnh
    print("🔍 Đang quét tập Val...")
    day_pool = []
    night_pool = []
    
    for item in labels:
        name = item['name']
        t_of_day = item.get('attributes', {}).get('timeofday', '')
        full_path = os.path.join(VAL_IMG_DIR, name)
        
        if os.path.exists(full_path):
            if t_of_day == 'daytime':
                day_pool.append(name)
            elif t_of_day == 'night':
                night_pool.append(name)

    # 2. In thống kê
    num_day = len(day_pool)
    num_night = len(night_pool)
    print(f"\nKẾT QUẢ THỐNG KÊ TẬP VAL:")
    print(f"   - Ảnh Ngày thực tế: {num_day}")
    print(f"   - Ảnh Đêm thực tế: {num_night}")

    # 3. Cân bằng
    min_val = min(num_day, num_night)
    print(f"⚖️  Quyết định cân bằng về mức: {min_val} ảnh mỗi bên.")

    selected_day = random.sample(day_pool, min_val)
    selected_night = random.sample(night_pool, min_val)

    # 4. Tạo Symlink (Xóa folder cũ nếu có để tránh trùng lặp)
    for sub in ['testA', 'testB']:
        target_dir = os.path.join(OUTPUT_PATH, sub)
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    for img in selected_day:
        os.symlink(os.path.join(VAL_IMG_DIR, img), os.path.join(OUTPUT_PATH, 'testA', img))
    for img in selected_night:
        os.symlink(os.path.join(VAL_IMG_DIR, img), os.path.join(OUTPUT_PATH, 'testB', img))

    print(f"\nĐã tạo xong tập Test cân bằng tại {OUTPUT_PATH}")
    print(f"Thư mục testA (Ngày): {min_val} ảnh")
    print(f"Thư mục testB (Đêm): {min_val} ảnh")

if __name__ == '__main__':
    run_balance()