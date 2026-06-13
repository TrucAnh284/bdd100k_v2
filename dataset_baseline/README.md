# YOLO11m Training - Baseline Subset (Day + Night Balanced)

Bộ dữ liệu này chứa sự kết hợp cân bằng giữa **ảnh Ngày (24,750)** và **ảnh Đêm (24,750)**.

## 1. Cài đặt môi trường (Setup)
Chạy các lệnh sau trên server:
```bash
# Cài đặt thư viện hệ thống cho OpenCV
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip

# Cài đặt Python requirements
pip install -r requirements.txt
```

## 2. Cấu hình đường dẫn (Important)
Trước khi chạy, hãy mở file `dataset.yaml` và kiểm tra dòng:
`path: /workspace/dataset_baseline`
Hãy đổi `/workspace` thành đường dẫn thực tế trên server của bạn.

## 3. Huấn luyện (Training)
```bash
python3 train_yolo11m_baseline.py
```
Kết quả sẽ được lưu tại: `runs/yolo11m_baseline_40ep/`

## 4. Đánh giá (Validation)
Sau khi train xong, chạy lệnh sau để kiểm tra mAP:
```bash
python3 val_yolo11m_baseline.py
```
*Lưu ý: File val này đã được thiết lập để loại bỏ lớp 'train' (ID 9) nhằm mang lại chỉ số khách quan hơn.*
