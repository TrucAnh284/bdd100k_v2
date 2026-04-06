import sys
import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# --- 1. SETUP HỆ THỐNG (Vượt qua bước Parser) ---
sys.argv = [
    'app.py', 
    '--dataroot', './datasets', 
    '--name', 'bdd_resnet9_10ep', 
    '--model', 'test', 
    '--no_dropout',
    '--preprocess', 'none',
    '--checkpoints_dir', '/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints'
]

try:
    from models.test_model import TestModel
    from options.test_options import TestOptions
except ImportError:
    st.error("Lỗi: Không tìm thấy module CycleGAN. Hãy kiểm tra thư mục chạy.")

# --- 2. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="AI Day-to-Night Converter", layout="wide")
st.title("🌙 BDD100K: Day to Night Image Translation")
st.markdown(f"**Sinh viên:** Elaine | **Thiết bị:** RTX 4090 | **Epoch:** 10")

# --- 3. HÀM NẠP MODEL (FIX LỖI WEIGHT TYPE) ---
@st.cache_resource
def load_ai_model():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # Cấu hình đường dẫn
    opt.checkpoints_dir = '/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints'
    opt.name = 'bdd_resnet9_10ep'
    opt.epoch = '4'  
    opt.device = torch.device('cuda:0')
    
    # Khởi tạo model
    model = TestModel(opt)
    
    # --- FIX LỖI CUDA TYPE ---
    # Ép toàn bộ mạng Generator lên GPU RTX 4090 ngay lập tức
    model.netG.to(opt.device)
    
    # Nạp trọng số từ file 10_net_G_A.pth
    load_path = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.epoch}_net_G_A.pth')
    
    if os.path.exists(load_path):
        state_dict = torch.load(load_path, map_location=str(opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # Nạp trọng số vào mạng đã nằm trên GPU
        model.netG.load_state_dict(state_dict)
    else:
        st.error(f"Không tìm thấy file: {load_path}")
        return None
        
    model.eval()
    return model

# --- 4. NẠP MODEL Ở SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Hệ thống")
    with st.spinner("Đang thắp đèn AI..."):
        model = load_ai_model()
    if model:
        st.success("AI đã sẵn sàng trên GPU!")

# --- 5. BỐ CỤC CHÍNH ---
col1, col2 = st.columns(2)

with col1:
    st.header("🖼️ Input (Day)")
    uploaded_file = st.file_uploader("Upload ảnh ban ngày...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh gốc ban ngày', use_container_width=True)

with col2:
    st.header("🌃 Output (Night)")
    if uploaded_file and model:
        if st.button('✨ Magic Sparkle! (Chuyển sang Đêm)'):
            with st.spinner('AI đang vẽ...'):
                try:
                    # Tiền xử lý (Resize & Normalize)
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    # Đưa ảnh đầu vào lên GPU (Cùng nơi với Model)
                    input_tensor = transform(image).unsqueeze(0).to(torch.device('cuda:0'))
                    
                    # Chạy Inference
                    with torch.no_grad():
                        output_tensor = model.netG(input_tensor)
                    
                    # Hậu xử lý (Tensor -> Image)
                    out = output_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
                    out = ((out + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                    
                    st.image(Image.fromarray(out), caption='Kết quả ban đêm (CycleGAN)', use_container_width=True)
                    st.balloons() # Ăn mừng thôi!
                
                except Exception as e:
                    st.error(f"Lỗi: {e}")

st.sidebar.info("Tips: Nếu ảnh bị mờ, có thể do resize 256x256. Đây là kích thước chuẩn model của Elaine.")