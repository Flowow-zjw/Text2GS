# Text2GS: 文本到3D高斯溅射

从文本描述自动生成 3D Gaussian Splatting 场景的端到端流水线。

## 📋 项目概述

Text2GS 整合了多个先进的生成模型，实现从文本到3D场景的完整生成流程：

```
文本提示 (Text Prompt)
    ↓
┌─────────────────────────────────────┐
│ Stage 1: MVDiffusion                │
│ 生成8张360°环绕全景视角图像          │
│ (每张45°间隔，覆盖完整环视)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: DUSt3R                     │
│ 从多视角图像重建稀疏点云             │
│ 估计相机内外参数                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 3: ViewCrafter                │
│ 在稀疏视角间插值生成稠密视角         │
│ 提升视角覆盖密度                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 4: 3D-GS Export               │
│ 导出COLMAP格式数据                   │
│ 可直接用于3D-GS训练                  │
└─────────────────────────────────────┘
```

## 🛠️ 环境要求

- Python 3.10
- CUDA 12.8
- GPU: 建议 24GB+ 显存 (如 RTX 4090, A100)
- 磁盘空间: 约 20GB (模型权重)

## 📦 安装部署

### 1. 克隆项目

```bash
git clone <https://github.com/Flowow-zjw/Text2GS>
cd Text2GS
```

### 2. 创建虚拟环境

```bash
conda create -n text2gs python=3.10
conda activate text2gs
```

### 3. 安装 PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 PyTorch3D

```bash
# 方式1: 直接安装 (推荐)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 方式2: 手动克隆编译
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..
```

### 6. 克隆外部依赖

```bash
# 创建 extern 目录
mkdir -p extern

# MVDiffusion
git clone https://github.com/Tangshitao/MVDiffusion.git ./extern/MVDiffusion

# ViewCrafter
git clone https://github.com/Drexubery/ViewCrafter.git ./extern/ViewCrafter

# DUSt3R
git clone --recursive https://github.com/naver/dust3r.git ./extern/dust3r
```

### 7. 下载模型权重

创建 checkpoints 目录结构：

```bash
mkdir -p checkpoints/mvdiffusion
mkdir -p checkpoints/viewcrafter
mkdir -p checkpoints/dust3r
```

下载以下模型：

| 模型 | 下载链接 | 保存路径 |
|------|----------|----------|
| MVDiffusion Panorama | [Dropbox](https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=0) | `checkpoints/mvdiffusion/pano.ckpt` |
| ViewCrafter Sparse | [HuggingFace](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt) | `checkpoints/viewcrafter/model_sparse.ckpt` |
| DUSt3R | [NaverLabs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | `checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |

命令行下载：

```bash
# MVDiffusion (需手动从Dropbox下载)

# ViewCrafter
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt -O checkpoints/viewcrafter/model_sparse.ckpt

# DUSt3R
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

## 🚀 使用方法

### 基础用法

```bash
python -m text2gs.run --text "A cozy living room with a fireplace and wooden furniture"
```

### 指定输出目录

```bash
python -m text2gs.run --text "..." --output ./my_output
```

## 📁 项目结构

```
Text2GS/
├── text2gs/                    # 核心代码
│   ├── __init__.py
│   ├── run.py                  # 命令行入口
│   ├── pipeline.py             # 流水线主逻辑
│   ├── stages/                 # 各阶段实现
│   │   ├── __init__.py
│   │   ├── base.py             # 基类
│   │   ├── mvdiffusion.py      # Stage 1: 多视角生成
│   │   ├── pointcloud.py       # Stage 2: 点云重建
│   │   ├── viewcrafter.py      # Stage 3: 稠密视角
│   │   └── gaussian.py         # Stage 4: 3D-GS导出
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── camera.py           # 相机参数处理
│       ├── render.py           # 渲染工具
│       └── io.py               # 文件读写
├── configs/                    # 配置文件
│   └── default.yaml
├── extern/                     # 外部依赖 (需手动克隆)
│   ├── MVDiffusion/
│   ├── ViewCrafter/
│   └── dust3r/
├── checkpoints/                # 模型权重 (需手动下载)
│   ├── mvdiffusion/
│   │   └── pano.ckpt
│   ├── viewcrafter/
│   │   └── model_sparse.ckpt
│   └── dust3r/
│       └── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
├── output/                     # 输出目录
├── requirements.txt
└── README.md
```

## 📤 输出结构

运行后会在 `output/<timestamp>/` 下生成：

```
output/20260101_120000/
├── stage1_mvdiffusion/         # Stage 1 输出
│   ├── view_00.png ~ view_07.png   # 8张全景视角
│   ├── cameras.npz             # 相机参数
│   ├── prompt.txt              # 输入提示
│   └── metadata.json
├── stage2_pointcloud/          # Stage 2 输出
│   ├── pointcloud.ply          # 稀疏点云
│   ├── images/                 # 输入图像
│   ├── depths/                 # 深度图
│   ├── cameras.npz             # 优化后相机参数
│   └── metadata.json
├── stage3_viewcrafter/         # Stage 3 输出
│   ├── videos/                 # 生成的视频
│   ├── frames/                 # 所有帧图像
│   ├── pointcloud.ply          # 更新的点云
│   ├── cameras.npz             # 插值相机参数
│   └── metadata.json
└── 3dgs/                       # Stage 4 输出 (COLMAP格式)
    ├── images/                 # 训练图像
    ├── sparse/0/               # COLMAP稀疏重建
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── colmap_output.txt
```


## 📚 引用

```bibtex
@inproceedings{tang2023mvdiffusion,
  title={MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion},
  author={Tang, Shitao and Zhang, Fuyang and Chen, Jiacheng and Wang, Peng and Furukawa, Yasutaka},
  booktitle={NeurIPS},
  year={2023}
}

@article{yu2024viewcrafter,
  title={ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis},
  author={Yu, Wangbo and Xing, Jinbo and Yuan, Li and Hu, Wenbo and Li, Xiaoyu and others},
  journal={TPAMI},
  year={2025}
}

@inproceedings{wang2024dust3r,
  title={DUSt3R: Geometric 3D Vision Made Easy},
  author={Wang, Shuzhe and Leroy, Vincent and Cabon, Yohann and Chidlovskii, Boris and Revaud, Jerome},
  booktitle={CVPR},
  year={2024}
}

@article{kerbl3Dgaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM TOG},
  year={2023}
}
```

## 📄 License

MIT License
