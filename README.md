Dự án sử dụng kiến trúc CycleGAN để thực hiện chuyển đổi hình ảnh từ miền Ngày (Day) sang Đêm (Night) dựa trên bộ dữ liệu BDD100K.

1. Môi trường thực nghiệm

GPU: NVIDIA RTX 4090 (24GB VRAM)
Framework: PyTorch, Torchvision
Môi trường: Docker (root@patient-orca)

2. Quy trình thực hiện nhanh

Bước 1:Chuẩn bị mã nguồn
# Di chuyển vào thư mục làm việc
cd /workspace

# Clone mã nguồn chính thức (Nếu chưa có)
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix

# Cài đặt thư viện cần thiết
# Bổ sung thư viện vào requirements
# --- Thư viện cốt lõi (AI Core) ---
torch==2.2.1
torchvision==0.17.1
torchaudio==2.2.1
numpy==1.26.4
Pillow==10.2.0
dominate==2.9.1
visdom==0.2.4

# --- Giao diện & Xử lý dữ liệu ---
streamlit==1.32.0
matplotlib==3.8.3
pandas==2.2.1
scipy==1.12.0
opencv-python==4.9.0.80

# --- Đánh giá & Tiện ích ---
pytorch-fid==0.3.0
gdown==5.1.0

# Tải thư viện
pip install -r requirements.txt

Bước 2: Tải dữ liệu từ Google Drive
# Tạo thư mục chứa dữ liệu gốc
mkdir -p /workspace/BDD100k_data
cd /workspace/BDD100k_data

# Tải file từ Drive (Yêu cầu gdown)
gdown 1R4mFPBzn526usqhOtLYpAK5mghRA9p9q

# Giải nén 
unzip BDD100k_data.zip

Bước 3: Tiền xử lý & Cân bằng dữ liệu (10k ảnh)
# Chạy script preprocess.py để lọc lấy 10,000 ảnh Ngày và 10,000 ảnh Đêm.

cd /workspace

python preprocess.py

Bước 4: Khởi chạy Visdom (Để theo dõi Loss)
# Chạy ngầm Visdom server
nohup python -m visdom.server > visdom.log 2>&1 &

3. Huấn luyện mô hình
# Chạy 10 Epoch với kiến trúc ResNet-9blocks và LSGAN (MSE Loss).

cd /workspace/pytorch-CycleGAN-and-pix2pix

python train.py \
  --dataroot /workspace/datasets/bdd_final \
  --name bdd_resnet9_10ep \
  --model cycle_gan \
  --netG resnet_9blocks \
  --batch_size 4 \
  --num_threads 16 \
  --save_epoch_freq 2 \
  --n_epochs 5 \
  --n_epochs_decay 0 \
  --no_html

4. Kiểm thử mô hình

python test.py \
  --dataroot /workspace/datasets/bdd_final \
  --name bdd_resnet9_10ep \
  --model cycle_gan \
  --no_dropout \
  --num_test 5000 \
  --results_dir ./results/bdd_final_test

5. Tính FID
# Trước khi tính thì  phân loại ảnh fake vào một folder riêng

python store_fake_img.py

# Tính FID
pip install pytorch-fid

python -m pytorch_fid \
  /workspace/datasets/bdd_final/testB \
  ./results/fake_B_only \
  --device cuda:0

6. Giao diện
# Dùng thư viện streamlit

pip install streamlit

streamlit run app.py
