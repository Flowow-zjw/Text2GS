# AutoDL 部署指南

## 环境信息
- **平台**: AutoDL
- **PyTorch**: 2.7.0
- **CUDA**: 12.8
- **推荐显存**: ≥24GB (A100/RTX 4090)

---

## 完整部署步骤

### Step 1: 创建实例

在 AutoDL 选择镜像：
- **基础镜像**: PyTorch 2.7.0 + CUDA 12.8
- **GPU**: A100-40G 或 RTX 4090 (推荐)
- **系统盘**: 50GB+
- **数据盘**: 100GB+ (存放模型权重)

---

### Step 2: 上传项目代码

```bash
# 方法1: 通过 AutoDL 文件管理上传 Text2GS 文件夹

# 方法2: 通过 git clone (如果有远程仓库)
cd /root/autodl-tmp
git clone <your-repo-url> Text2GS
cd Text2GS
```

---

### Step 3: 克隆外部依赖

```bash
cd /root/autodl-tmp/Text2GS

# 创建 extern 目录
mkdir -p extern

# 克隆 MVDiffusion
git clone https://github.com/Tangshitao/MVDiffusion.git extern/MVDiffusion

# 克隆 ViewCrafter
git clone https://github.com/Drexubery/ViewCrafter.git extern/ViewCrafter

# 克隆 DUSt3R (递归克隆)
git clone --recursive https://github.com/naver/dust3r.git extern/dust3r
```

---

### Step 4: 安装依赖

```bash
# 基础依赖 (PyTorch 已预装，跳过)
pip install numpy==1.24.0
pip install einops==0.6.1
pip install omegaconf==2.3.0
pip install pyyaml
pip install tqdm

# Diffusion 相关
pip install diffusers==0.21.0
pip install transformers==4.35.0
pip install accelerate
pip install open-clip-torch==2.17.1
pip install timm==0.6.13

# 图像/视频处理
pip install opencv-python==4.8.0.74
pip install Pillow==10.0.0
pip install imageio==2.31.1
pip install imageio-ffmpeg
pip install av
pip install decord

# 3D 相关
pip install trimesh
pip install roma
pip install kornia
pip install scipy
pip install scikit-image
pip install scikit-learn
pip install matplotlib

# PyTorch3D (CUDA 12.x 版本)
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# 如果上面失败，从源码编译
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# DUSt3R 依赖
cd extern/dust3r
pip install -r requirements.txt
cd ../..

# 安装 xformers (加速)
pip install xformers
```

---

### Step 5: 下载模型权重

```bash
cd /root/autodl-tmp/Text2GS

# 创建目录
mkdir -p checkpoints/dust3r
mkdir -p checkpoints/viewcrafter
mkdir -p checkpoints/mvdiffusion

# 1. 下载 DUSt3R (~1GB)
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# 2. 下载 ViewCrafter Sparse (~10GB) - 多视图版本
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt \
    -O checkpoints/viewcrafter/model_sparse.ckpt

# 3. MVDiffusion 权重需要手动下载
# 访问: https://github.com/Tangshitao/MVDiffusion/releases
# 下载 pano.ckpt 并上传到 checkpoints/mvdiffusion/pano.ckpt
```

**MVDiffusion 权重下载方法**:
```bash
# 如果有直接链接
wget <mvdiffusion_download_url> -O checkpoints/mvdiffusion/pano.ckpt

# 或者通过 AutoDL 文件管理上传
```

---

### Step 6: 验证安装

```bash
cd /root/autodl-tmp/Text2GS

# 验证 PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 验证 PyTorch3D
python -c "import pytorch3d; print('PyTorch3D: OK')"

# 验证 DUSt3R
python -c "
import sys
sys.path.insert(0, './extern/dust3r')
from dust3r.inference import load_model
print('DUSt3R: OK')
"

# 验证权重文件
ls -lh checkpoints/*/
```

---

### Step 7: 运行测试

```bash
cd /root/autodl-tmp/Text2GS

# 安装项目
pip install -e .

# 运行
python -m text2gs.run \
    --text "A cozy living room with a fireplace and wooden furniture" \
    --output /root/autodl-tmp/output \
    --device cuda:0
```

---

## 一键部署脚本

将以下内容保存为 `setup_autodl.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Text2GS AutoDL 部署脚本"
echo "=========================================="

cd /root/autodl-tmp

# 1. 克隆外部依赖
echo "[1/5] 克隆外部依赖..."
mkdir -p Text2GS/extern
cd Text2GS/extern

if [ ! -d "MVDiffusion" ]; then
    git clone https://github.com/Tangshitao/MVDiffusion.git
fi

if [ ! -d "ViewCrafter" ]; then
    git clone https://github.com/Drexubery/ViewCrafter.git
fi

if [ ! -d "dust3r" ]; then
    git clone --recursive https://github.com/naver/dust3r.git
fi

cd ..

# 2. 安装依赖
echo "[2/5] 安装 Python 依赖..."
pip install numpy==1.24.0 einops==0.6.1 omegaconf==2.3.0 pyyaml tqdm -q
pip install diffusers==0.21.0 transformers==4.35.0 accelerate -q
pip install open-clip-torch==2.17.1 timm==0.6.13 -q
pip install opencv-python Pillow imageio imageio-ffmpeg av decord -q
pip install trimesh roma kornia scipy scikit-image scikit-learn matplotlib -q

# PyTorch3D
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html -q || \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" -q

# DUSt3R 依赖
pip install -r extern/dust3r/requirements.txt -q

# xformers
pip install xformers -q

# 3. 下载权重
echo "[3/5] 下载模型权重..."
mkdir -p checkpoints/dust3r checkpoints/viewcrafter checkpoints/mvdiffusion

if [ ! -f "checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
        -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
fi

if [ ! -f "checkpoints/viewcrafter/model_sparse.ckpt" ]; then
    wget -q --show-progress https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt \
        -O checkpoints/viewcrafter/model_sparse.ckpt
fi

# 4. 安装项目
echo "[4/5] 安装 Text2GS..."
pip install -e . -q

# 5. 验证
echo "[5/5] 验证安装..."
python -c "
import torch
import pytorch3d
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('PyTorch3D: OK')
print('Installation successful!')
"

echo ""
echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "注意: MVDiffusion 权重需要手动下载"
echo "  1. 访问 https://github.com/Tangshitao/MVDiffusion/releases"
echo "  2. 下载 pano.ckpt"
echo "  3. 上传到 checkpoints/mvdiffusion/pano.ckpt"
echo ""
echo "运行命令:"
echo "  python -m text2gs.run --text \"Your prompt\" --output ./output"
```

---

## 常见问题

### Q1: PyTorch3D 安装失败

```bash
# 方法1: 使用预编译 wheel (推荐)
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# 方法2: 从源码编译
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 方法3: conda 安装
conda install pytorch3d -c pytorch3d
```

### Q2: CUDA Out of Memory

```bash
# 修改配置减少显存使用
# 编辑 configs/default.yaml

viewcrafter:
  video_length: 16  # 从 25 减少到 16
  ddim_steps: 30    # 从 50 减少到 30
```

### Q3: 下载速度慢

```bash
# 使用 AutoDL 学术加速
source /etc/network_turbo

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q4: DUSt3R 导入错误

```bash
# 确保路径正确
export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/Text2GS/extern/dust3r"
```

---

## 目录结构

部署完成后的目录结构：

```
/root/autodl-tmp/Text2GS/
├── text2gs/                    # 核心代码
├── configs/
│   └── default.yaml
├── extern/                     # 外部依赖
│   ├── MVDiffusion/
│   ├── ViewCrafter/
│   └── dust3r/
├── checkpoints/               # 模型权重
│   ├── dust3r/
│   │   └── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth  (~1GB)
│   ├── viewcrafter/
│   │   └── model_sparse.ckpt                        (~10GB)
│   └── mvdiffusion/
│       └── pano.ckpt                                (~5GB, 手动下载)
├── output/                    # 输出目录
├── setup.py
└── requirements.txt
```

---

## 运行示例

```bash
# 基础运行
python -m text2gs.run \
    --text "A modern kitchen with white cabinets and marble countertops" \
    --output /root/autodl-tmp/output

# 指定配置
python -m text2gs.run \
    --config configs/default.yaml \
    --text "A bedroom with a large bed and soft lighting" \
    --output /root/autodl-tmp/output

# 节省显存模式
python -m text2gs.run \
    --text "A cozy cafe with wooden tables" \
    --output /root/autodl-tmp/output \
    --unload-between-stages
```
