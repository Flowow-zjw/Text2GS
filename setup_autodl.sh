#!/bin/bash
# Text2GS AutoDL 一键部署脚本
# PyTorch 2.7.0 + CUDA 12.8

set -e

echo "=========================================="
echo "Text2GS AutoDL 部署脚本"
echo "PyTorch 2.7.0 + CUDA 12.8"
echo "=========================================="

# 设置工作目录
WORK_DIR="/root/autodl-tmp/Text2GS"
cd $WORK_DIR

# 开启学术加速 (如果可用)
if [ -f "/etc/network_turbo" ]; then
    source /etc/network_turbo
    echo "学术加速已开启"
fi

# ==================== Step 1: 克隆外部依赖 ====================
echo ""
echo "[1/6] 克隆外部依赖..."

mkdir -p extern
cd extern

# MVDiffusion
if [ ! -d "MVDiffusion" ]; then
    echo "  克隆 MVDiffusion..."
    git clone --depth 1 https://github.com/Tangshitao/MVDiffusion.git
else
    echo "  MVDiffusion 已存在"
fi

# ViewCrafter
if [ ! -d "ViewCrafter" ]; then
    echo "  克隆 ViewCrafter..."
    git clone --depth 1 https://github.com/Drexubery/ViewCrafter.git
else
    echo "  ViewCrafter 已存在"
fi

# DUSt3R
if [ ! -d "dust3r" ]; then
    echo "  克隆 DUSt3R..."
    git clone --recursive --depth 1 https://github.com/naver/dust3r.git
else
    echo "  DUSt3R 已存在"
fi

cd $WORK_DIR

# ==================== Step 2: 安装基础依赖 ====================
echo ""
echo "[2/6] 安装基础依赖..."

pip install --upgrade pip -q

# 核心依赖
pip install numpy==1.24.0 -q
pip install einops==0.6.1 -q
pip install omegaconf==2.3.0 -q
pip install pyyaml tqdm -q

# Diffusion 相关 (兼容 PyTorch 2.7)
pip install diffusers==0.27.0 -q
pip install transformers==4.40.0 -q
pip install accelerate -q
pip install safetensors -q

# CLIP 和 timm
pip install open-clip-torch -q
pip install timm==0.9.12 -q

echo "  基础依赖安装完成"

# ==================== Step 3: 安装图像/视频/3D依赖 ====================
echo ""
echo "[3/6] 安装图像/视频/3D依赖..."

# 图像视频处理
pip install opencv-python -q
pip install Pillow -q
pip install imageio imageio-ffmpeg -q
pip install av decord -q

# 3D 相关
pip install trimesh -q
pip install roma -q
pip install kornia -q
pip install scipy -q
pip install scikit-image scikit-learn -q
pip install matplotlib -q

# PyTorch3D (CUDA 12.x)
echo "  安装 PyTorch3D..."
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html -q 2>/dev/null || {
    echo "  预编译版本失败，尝试从源码编译..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" -q
}

# xformers (加速)
pip install xformers -q 2>/dev/null || echo "  xformers 安装失败，跳过"

echo "  图像/视频/3D依赖安装完成"

# ==================== Step 4: 安装 DUSt3R 依赖 ====================
echo ""
echo "[4/6] 安装 DUSt3R 依赖..."

cd extern/dust3r
pip install -r requirements.txt -q 2>/dev/null || echo "  部分 DUSt3R 依赖已安装"
cd $WORK_DIR

echo "  DUSt3R 依赖安装完成"

# ==================== Step 5: 下载模型权重 ====================
echo ""
echo "[5/6] 下载模型权重..."

mkdir -p checkpoints/dust3r
mkdir -p checkpoints/viewcrafter
mkdir -p checkpoints/mvdiffusion

# DUSt3R 权重 (~1GB)
if [ ! -f "checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "  下载 DUSt3R 权重 (~1GB)..."
    wget -q --show-progress \
        https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
        -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
else
    echo "  DUSt3R 权重已存在"
fi

# ViewCrafter Sparse 权重 (~10GB)
if [ ! -f "checkpoints/viewcrafter/model_sparse.ckpt" ]; then
    echo "  下载 ViewCrafter Sparse 权重 (~10GB)..."
    wget -q --show-progress \
        https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt \
        -O checkpoints/viewcrafter/model_sparse.ckpt
else
    echo "  ViewCrafter 权重已存在"
fi

# MVDiffusion 权重检查
if [ ! -f "checkpoints/mvdiffusion/pano.ckpt" ]; then
    echo ""
    echo "  ⚠️  MVDiffusion 权重需要手动下载!"
    echo "  请访问: https://github.com/Tangshitao/MVDiffusion/releases"
    echo "  下载 pano.ckpt 并放到: checkpoints/mvdiffusion/pano.ckpt"
else
    echo "  MVDiffusion 权重已存在"
fi

# ==================== Step 6: 安装项目 ====================
echo ""
echo "[6/6] 安装 Text2GS..."

pip install -e . -q

# ==================== 验证安装 ====================
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python -c "
import sys
import torch

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

try:
    import pytorch3d
    print('PyTorch3D: ✓')
except:
    print('PyTorch3D: ✗')

try:
    sys.path.insert(0, './extern/dust3r')
    from dust3r.inference import load_model
    print('DUSt3R: ✓')
except Exception as e:
    print(f'DUSt3R: ✗ ({e})')

try:
    import diffusers
    print(f'Diffusers: ✓ ({diffusers.__version__})')
except:
    print('Diffusers: ✗')
"

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "权重状态:"
ls -lh checkpoints/*/ 2>/dev/null || echo "  无权重文件"
echo ""

if [ ! -f "checkpoints/mvdiffusion/pano.ckpt" ]; then
    echo "⚠️  注意: MVDiffusion 权重需要手动下载"
    echo "   1. 访问 https://github.com/Tangshitao/MVDiffusion/releases"
    echo "   2. 下载 pano.ckpt"
    echo "   3. 上传到 $WORK_DIR/checkpoints/mvdiffusion/pano.ckpt"
    echo ""
fi

echo "运行命令:"
echo "  cd $WORK_DIR"
echo "  python -m text2gs.run --text \"A cozy living room\" --output ./output"
echo ""
