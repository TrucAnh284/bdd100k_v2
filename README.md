Dự án sử dụng kiến trúc CycleGAN để thực hiện chuyển đổi hình ảnh từ miền Ngày (Day) sang Đêm (Night) dựa trên bộ dữ liệu BDD100K.

# 1. Môi trường thực nghiệm

GPU: NVIDIA RTX 4090 (24GB VRAM)
Framework: PyTorch, Torchvision
Môi trường: Docker (root@patient-orca)

# 2. Quy trình thực hiện nhanh

#Bước 1:Chuẩn bị mã nguồn
<small># Di chuyển vào thư mục làm việc<small>
cd /workspace

<small># Clone mã nguồn chính thức (Nếu chưa có)<small>
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix

<small># Cài đặt thư viện cần thiết<small>
<small># Bổ sung thư viện vào requirements<small>
<small># --- Thư viện cốt lõi (AI Core) ---<small>
torch==2.2.1
torchvision==0.17.1
torchaudio==2.2.1
numpy==1.26.4
Pillow==10.2.0
dominate==2.9.1
visdom==0.2.4

<small># --- Giao diện & Xử lý dữ liệu ---<small>
streamlit==1.32.0
matplotlib==3.8.3
pandas==2.2.1
scipy==1.12.0
opencv-python==4.9.0.80

<small># --- Đánh giá & Tiện ích --- <small>
pytorch-fid==0.3.0
gdown==5.1.0

<small># Tải thư viện<small>
pip install -r requirements.txt

# Bước 2: Tải dữ liệu từ Google Drive
<small> Tạo thư mục chứa dữ liệu gốc <small>
mkdir -p /workspace/BDD100k_data
cd /workspace/BDD100k_data

<small># Tải file từ Drive (Yêu cầu gdown) <small>
gdown 1R4mFPBzn526usqhOtLYpAK5mghRA9p9q

<small># Giải nén <small> 
unzip BDD100k_data.zip

# Bước 3: Tiền xử lý & Cân bằng dữ liệu (10k ảnh)
cd /workspace

python preprocess.py

# Bước 4: Khởi chạy Visdom (Để theo dõi Loss)

nohup python -m visdom.server > visdom.log 2>&1 &

# 3. Huấn luyện mô hình
<small># Chạy training với kiến trúc ResNet-9blocks và LSGAN (MSE Loss)</small>

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

# 4. Kiểm thử mô hình

python test.py \
  --dataroot /workspace/datasets/bdd_final \
  --name bdd_resnet9_10ep \
  --model cycle_gan \
  --no_dropout \
  --num_test 5000 \
  --results_dir ./results/bdd_final_test

# 5. Tính FID
<small># Sinh ảnh test và tính toán chỉ số FID</small>

python store_fake_img.py

pip install pytorch-fid

python -m pytorch_fid \
  /workspace/datasets/bdd_final/testB \
  ./results/fake_B_only \
  --device cuda:0

# 6. Giao diện
<small># Sinh ảnh test và tính toán chỉ số FID</small>

pip install streamlit

streamlit run app.py
