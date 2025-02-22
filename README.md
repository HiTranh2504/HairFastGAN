# HairFastGAN - Virtual Hair Style

HairFastGAN là một mô hình AI giúp thay đổi kiểu tóc và màu tóc trong ảnh chân dung một cách nhanh chóng và chính xác bằng cách sử dụng công nghệ Generative Adversarial Networks (GANs).

---
## 1. Cài đặt

### Yêu cầu hệ thống
- Python >= 3.8
- CUDA (khuyến nghị để chạy nhanh hơn trên GPU)
- Các thư viện: `torch`, `torchvision`, `streamlit`, `face_alignment`, `dill`, `pillow`, `addict`, `pyngrok`

### Hướng dẫn cài đặt

#### **Bước 1: Clone repository**
```sh
!git clone https://github.com/AIRI-Institute/HairFastGAN.git
%cd HairFastGAN
```

#### **Bước 2: Cài đặt Ninja** (hệ thống build)
```sh
!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

#### **Bước 3: Cài đặt thư viện phụ thuộc**
```sh
!pip install pyngrok streamlit pillow==10.0.0 face_alignment dill==0.2.7.1 addict fpie git+https://github.com/openai/CLIP.git -q
!pip install torchvision
```

#### **Bước 4: Tải mô hình đã được huấn luyện**
```sh
!git clone https://huggingface.co/AIRI-Institute/HairFastGAN
!cd HairFastGAN && git lfs pull && cd ..
!mv HairFastGAN/pretrained_models pretrained_models
!mv HairFastGAN/input input
!rm -rf HairFastGAN
```

---
## 2. Cách sử dụng

### **Chạy mô hình HairFastGAN**

#### **Bước 1: Khởi tạo mô hình**
```python
from pathlib import Path
from hair_swap import HairFast, get_parser
import torch

model_args = get_parser()
hair_fast = HairFast(model_args.parse_args([]))
```

#### **Bước 2: Swap kiểu tóc và màu tóc**
```python
input_dir = Path("HairFastGAN/Images")
face_path = input_dir / 'LISA.jpg'
shape_path = input_dir / 'jisoo.png'
color_path = input_dir / 'ROSE.png'

final_image, face_align, shape_align, color_align = hair_fast.swap(face_path, shape_path, color_path, align=True)
```

#### **Bước 3: Hiển thị kết quả**
```python
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

def display_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
    for ax, (title, img) in zip(axes, images.items()):
        img = T.functional.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        ax.imshow(img)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    plt.show()

display_images({"Final Result": final_image, "Input Face": face_align, "Hair Shape": shape_align, "Hair Color": color_align})
```

---
## 3. Chạy ứng dụng Streamlit

Dự án này cung cấp giao diện Streamlit để người dùng dễ dàng thay đổi kiểu tóc.

### **Chạy ứng dụng**
```sh
streamlit run app.py
```

### **Giao diện Web**
Ứng dụng sẽ mở một giao diện web, nơi bạn có thể:
1. **Tải ảnh khuôn mặt của bạn**
2. **Chọn ảnh mẫu tóc**
3. **Chọn ảnh màu tóc**
4. **Nhấn nút "Change me!!!" để xem kết quả**

---
## 4. Đóng góp
Nếu bạn muốn cải thiện HairFastGAN, hãy fork repository và gửi pull request. Mọi đóng góp đều được hoan nghênh!

---
## 5. Tài nguyên
- **Paper:** [HairFastGAN: High-Speed Hair Style Transfer using GANs](https://arxiv.org/abs/xxxxxx)
- **Repository chính thức:** [AIRI-Institute/HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN)
- **Mô hình đã huấn luyện:** [Hugging Face Model Hub](https://huggingface.co/AIRI-Institute/HairFastGAN)

---
## 6. Liên hệ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ qua email: `support@hairfastgan.com`.

