# Text2GS: Text-to-3D Gaussian Splatting

从文本描述生成 3D Gaussian Splatting 场景。

## 流程

```
Text Prompt
    ↓
MVDiffusion (8张360°全景视角)
    ↓
DUSt3R (点云重建)
    ↓
ViewCrafter (稠密视角生成)
    ↓
3D-GS (高斯溅射训练)
```

## 安装

```bash
# 1. 克隆项目
git clone <repo_url>
cd Text2GS

# 2. 初始化子模块
git submodule update --init --recursive

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型权重
python scripts/download_weights.py
```

## 使用

```bash
# 基础用法
python -m text2gs.run --text "A cozy living room with a fireplace"

# 指定输出目录
python -m text2gs.run --text "..." --output ./output

# 使用配置文件
python -m text2gs.run --config configs/default.yaml --text "..."
```

## 项目结构

```
Text2GS/
├── text2gs/                 # 核心代码
│   ├── __init__.py
│   ├── run.py              # 主入口
│   ├── pipeline.py         # 完整流程
│   ├── stages/             # 各阶段实现
│   │   ├── __init__.py
│   │   ├── mvdiffusion.py  # Stage 1
│   │   ├── pointcloud.py   # Stage 2
│   │   ├── viewcrafter.py  # Stage 3
│   │   └── gaussian.py     # Stage 4
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── camera.py
│       ├── render.py
│       └── io.py
├── configs/                 # 配置文件
│   └── default.yaml
├── scripts/                 # 脚本
│   ├── download_weights.py
│   └── setup_env.py
├── extern/                  # 外部依赖 (子模块)
│   ├── MVDiffusion/
│   ├── ViewCrafter/
│   └── dust3r/
├── checkpoints/            # 模型权重
├── output/                 # 输出目录
├── requirements.txt
└── README.md
```

## 引用

```bibtex
@inproceedings{tang2023mvdiffusion,
  title={MVDiffusion},
  author={Tang, Shitao and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{yu2024viewcrafter,
  title={ViewCrafter},
  author={Yu, Wangbo and others},
  year={2024}
}
```
