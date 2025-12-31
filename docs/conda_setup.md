# Conda 手动环境搭建指南

## 环境要求
- **Conda**: Miniconda 或 Anaconda
- **CUDA**: 12.1 / 12.4 / 12.8 (AutoDL 预装)
- **GPU**: ≥24GB 显存 (A100/RTX 4090)

---

## Step 1: 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n text2gs python=3.10 -y

# 激活环境
conda activate text2gs
```

---

## Step 2: 安装 PyTorch (CUDA 12.x)

```bash
# PyTorch 2.3.0 + CUDA 12.1 (推荐，稳定)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 或者 PyTorch 2.4.0 + CUDA 12.4
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Step 3: 安装 PyTorch3D

```bash
# 方法1: 预编译 wheel (推荐)
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

# 方法2: 如果方法1失败，从源码编译
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 方法3: conda 安装
# conda install pytorch3d -c pytorch3d

# 验证
python -c "import pytorch3d; print('PyTorch3D: OK')"
```

---

## Step 4: 安装核心依赖

```bash
# 基础库
pip install numpy==1.24.0
pip install einops==0.6.1
pip install omegaconf==2.3.0
pip install pyyaml
pip install tqdm

# Diffusion 模型
pip install diffusers==0.27.0
pip install transformers==4.40.0
pip install accelerate
pip install safetensors

# CLIP 和 Vision
pip install open-clip-torch
pip install timm==0.9.12
```

---

## Step 5: 安装图像/视频处理库

```bash
# 图像处理
pip install opencv-python
pip install Pillow
pip install scikit-image

# 视频处理
pip install imageio
pip install imageio-ffmpeg
pip install av
pip install decord
```

---

## Step 6: 安装 3D 相关库

```bash
# 3D 处理
pip install trimesh
pip install roma
pip install kornia

# 科学计算
pip install scipy
pip install scikit-learn
pip install matplotlib
```

---

## Step 7: 安装 xformers (可选，加速)

```bash
# xformers 加速注意力计算
pip install xformers

# 验证
python -c "import xformers; print('xformers: OK')"
```

---

## Step 8: 克隆外部依赖

```bash
cd /root/autodl-tmp/Text2GS  # 或你的项目目录

# 创建 extern 目录
mkdir -p extern
cd extern

# 克隆 MVDiffusion
git clone --depth 1 https://github.com/Tangshitao/MVDiffusion.git

# 克隆 ViewCrafter
git clone --depth 1 https://github.com/Drexubery/ViewCrafter.git

# 克隆 DUSt3R (递归)
git clone --recursive --depth 1 https://github.com/naver/dust3r.git

cd ..
```

---

## Step 9: 安装 DUSt3R 依赖

```bash
cd extern/dust3r

# 安装依赖
pip install -r requirements.txt

# 返回项目根目录
cd ../..
```

---

## Step 10: 下载模型权重

```bash
# 创建目录
mkdir -p checkpoints/dust3r
mkdir -p checkpoints/viewcrafter
mkdir -p checkpoints/mvdiffusion

# 1. DUSt3R (~1GB)
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# 2. ViewCrafter Sparse (~10GB) - 多视图版本
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt \
    -O checkpoints/viewcrafter/model_sparse.ckpt

# 3. MVDiffusion - 手动下载
# 访问: https://github.com/Tangshitao/MVDiffusion/releases
# 下载 pano.ckpt 放到 checkpoints/mvdiffusion/pano.ckpt
```

---

## Step 11: 安装项目

```bash
# 安装 Text2GS
pip install -e .
```

---

## Step 12: 验证完整安装

```bash
python -c "
import sys
import torch

print('='*50)
print('环境验证')
print('='*50)

# Python
print(f'Python: {sys.version.split()[0]}')

# PyTorch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {mem:.1f} GB')

# PyTorch3D
try:
    import pytorch3d
    print('PyTorch3D: ✓')
except ImportError as e:
    print(f'PyTorch3D: ✗ ({e})')

# Diffusers
try:
    import diffusers
    print(f'Diffusers: ✓ ({diffusers.__version__})')
except ImportError:
    print('Diffusers: ✗')

# DUSt3R
try:
    sys.path.insert(0, './extern/dust3r')
    from dust3r.inference import load_model
    print('DUSt3R: ✓')
except Exception as e:
    print(f'DUSt3R: ✗ ({e})')

# xformers
try:
    import xformers
    print('xformers: ✓')
except ImportError:
    print('xformers: ✗ (可选)')

print('='*50)
"
```

---

## 完整命令汇总

```bash
# ========== 一次性复制执行 ==========

# 1. 创建环境
conda create -n text2gs python=3.10 -y
conda activate text2gs

# 2. PyTorch + CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. PyTorch3D
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

# 4. 核心依赖
pip install numpy==1.24.0 einops==0.6.1 omegaconf==2.3.0 pyyaml tqdm
pip install diffusers==0.27.0 transformers==4.40.0 accelerate safetensors
pip install open-clip-torch timm==0.9.12

# 5. 图像/视频/3D
pip install opencv-python Pillow scikit-image
pip install imageio imageio-ffmpeg av decord
pip install trimesh roma kornia scipy scikit-learn matplotlib

# 6. 可选加速
pip install xformers

# 7. 克隆依赖
mkdir -p extern && cd extern
git clone --depth 1 https://github.com/Tangshitao/MVDiffusion.git
git clone --depth 1 https://github.com/Drexubery/ViewCrafter.git
git clone --recursive --depth 1 https://github.com/naver/dust3r.git
cd ..

# 8. DUSt3R 依赖
pip install -r extern/dust3r/requirements.txt

# 9. 下载权重
mkdir -p checkpoints/{dust3r,viewcrafter,mvdiffusion}
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt -O checkpoints/viewcrafter/model_sparse.ckpt

# 10. 安装项目
pip install -e .
```

---

## 运行测试

```bash
# 激活环境
conda activate text2gs

# 运行
python -m text2gs.run \
    --text "A cozy living room with a fireplace and wooden furniture" \
    --output ./output \
    --device cuda:0
```

---

## 常见问题

### Q1: PyTorch3D 安装失败

```bash
# 检查 PyTorch 版本对应的 wheel
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# PyTorch 2.3.0 + CUDA 12.1 + Python 3.10
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html

# PyTorch 2.1.0 + CUDA 12.1 + Python 3.10
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
```

### Q2: CUDA 版本不匹配

```bash
# 检查系统 CUDA
nvcc --version
nvidia-smi

# 安装对应版本 PyTorch
# CUDA 11.8
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### Q3: 显存不足

```bash
# 编辑 configs/default.yaml
viewcrafter:
  video_length: 16      # 减少帧数
  ddim_steps: 30        # 减少推理步数

# 或使用 --unload-between-stages 参数
python -m text2gs.run --text "..." --unload-between-stages
```

### Q4: 下载速度慢

```bash
# AutoDL 学术加速
source /etc/network_turbo

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```
