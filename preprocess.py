import os
import random

# Đường dẫn đến folder có sẵn (dựa trên ảnh của Elaine)
SOURCE_BASE = '/workspace/BDD100k_data/BDD100k_data/images/train'
TARGET_BASE = '/workspace/datasets/bdd_10k_balanced'
LIMIT = 10000

def quick_balance():
    for sub in ['trainA', 'trainB', 'testA', 'testB']:
        src_dir = os.path.join(SOURCE_BASE, sub)
        dst_dir = os.path.join(TARGET_BASE, sub)
        os.makedirs(dst_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            print(f"⚠️ Không tìm thấy {src_dir}, bỏ qua...")
            continue
            
        files = os.listdir(src_dir)
        random.shuffle(files)
        
        # Chỉ lấy tối đa 10k cho tập train, tập test giữ nguyên hoặc lấy hết
        selected = files[:LIMIT] if 'train' in sub else files
        
        for img in selected:
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
        
        print(f"✅ Đã tạo symlink cho {len(selected)} ảnh vào {sub}")

quick_balance()