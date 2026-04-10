# Text2GS: Text-to-3D Gaussian Splatting

An end-to-end pipeline for generating 3D Gaussian Splatting scenes from text descriptions.

## 📋 Overview

Existing text-to-3D generation methods face several challenges:
- Single-view generation struggles to maintain multi-view consistency
- Direct optimization of NeRF/3D-GS is time-consuming and prone to local optima
- Lack of effective support for 360° panoramic views of indoor scenes

Text2GS uses a **progressive multi-stage generation framework** that decomposes the complex text-to-3D task into controllable sub-tasks, progressively building high-quality 3D scenes.

### Key Contributions

1. **Multi-view Consistent Generation**: Leverages MVDiffusion's correspondence-aware mechanism to explicitly model geometric constraints between views during diffusion, generating globally consistent 360° panoramic images

2. **Geometry-Appearance Decoupled Reconstruction**: Employs DUSt3R for calibration-free dense point cloud reconstruction, separating geometry estimation from appearance generation to improve reconstruction robustness

3. **Video Diffusion-Driven View Interpolation**: Applies ViewCrafter video diffusion model for sparse view interpolation, utilizing temporal consistency of video generation to ensure smooth transitions between interpolated views

4. **End-to-End Automated Pipeline**: Designs a modular four-stage pipeline supporting intermediate result visualization and parameter tuning for analysis and improvement

### Pipeline

```
Text ──→ MVDiffusion ──→ DUSt3R ──→ ViewCrafter ──→ 3D-GS
         (Multi-view)    (Point Cloud) (Interpolation) (Scene)
```

| Stage | Input | Output | Key Techniques |
|-------|-------|--------|----------------|
| Stage 1 | Text prompt | 8 panoramic images | Correspondence-aware diffusion |
| Stage 2 | Multi-view images | Point cloud + poses | Calibration-free 3D reconstruction |
| Stage 3 | Sparse views + point cloud | Dense view sequence | Video diffusion, point cloud rendering |
| Stage 4 | Dense views + poses | 3D-GS scene | COLMAP format export + Optional training |

## 🛠️ Requirements

- Python 3.10
- CUDA 12.8
- GPU: 24GB+ VRAM recommended (e.g., RTX 4090, A100)
- Disk space: ~20GB (model weights)

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Flowow-zjw/Text2GS
cd Text2GS
```

### 2. Create virtual environment

```bash
conda create -n text2gs python=3.10
conda activate text2gs
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install PyTorch3D

```bash
# Option 1: Direct installation (recommended)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Option 2: Manual build
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..
```

### 6. Clone external dependencies

```bash
# Create extern directory
mkdir -p extern

# MVDiffusion
git clone https://github.com/Tangshitao/MVDiffusion.git ./extern/MVDiffusion

# ViewCrafter
git clone https://github.com/Drexubery/ViewCrafter.git ./extern/ViewCrafter

# DUSt3R
git clone --recursive https://github.com/naver/dust3r.git ./extern/dust3r
```

### 7. Download model weights

Create checkpoint directories:

```bash
mkdir -p checkpoints/mvdiffusion
mkdir -p checkpoints/viewcrafter
mkdir -p checkpoints/dust3r
```

Download the following models:

| Model | Download Link | Save Path |
|-------|---------------|-----------|
| MVDiffusion Panorama | [Dropbox](https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=0) | `checkpoints/mvdiffusion/pano.ckpt` |
| ViewCrafter Sparse | [HuggingFace](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt) | `checkpoints/viewcrafter/model_sparse.ckpt` |
| DUSt3R | [NaverLabs](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | `checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |

Command line download:

```bash
# MVDiffusion (manual download from Dropbox required)

# ViewCrafter
wget https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt -O checkpoints/viewcrafter/model_sparse.ckpt

# DUSt3R
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

## 🚀 Usage

### Basic usage (Export only)

```bash
python -m text2gs.run --text "A cozy living room with a fireplace and wooden furniture"
```

### With 3D-GS Training (Integrated)

```bash
# Train with default 30000 iterations
python -m text2gs.run \
  --text "A modern kitchen with marble countertops" \
  --train-3dgs

# Custom iterations
python -m text2gs.run \
  --text "A beautiful garden" \
  --train-3dgs \
  --gs-iterations 50000

# With memory optimization
python -m text2gs.run \
  --text "A spacious library" \
  --train-3dgs \
  --unload-between-stages
```

### Specify output directory

```bash
python -m text2gs.run --text "..." --output ./my_output
```

For detailed training guide, see [TRAINING_GUIDE.md](./TRAINING_GUIDE.md).

## 📚 Documentation

- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - Complete training guide with examples
- [OUTPUT_STRUCTURE.md](./OUTPUT_STRUCTURE.md) - Detailed output structure explanation
- [SAVE_RESULTS_SUMMARY.md](./SAVE_RESULTS_SUMMARY.md) - Result saving features
- [INTEGRATION_SUMMARY.md](./INTEGRATION_SUMMARY.md) - Integration summary
- [CHANGELOG.md](./CHANGELOG.md) - Version history and updates

## 🔍 Verify Results

After running the pipeline, verify output completeness:

```bash
python check_output.py output/TIMESTAMP
```

This will check all stages and report any missing files.

## 📁 Project Structure

```
Text2GS/
├── text2gs/                    # Core code
│   ├── __init__.py
│   ├── run.py                  # CLI entry point
│   ├── pipeline.py             # Pipeline logic
│   ├── stages/                 # Stage implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base class
│   │   ├── mvdiffusion.py      # Stage 1: Multi-view generation
│   │   ├── pointcloud.py       # Stage 2: Point cloud reconstruction
│   │   ├── viewcrafter.py      # Stage 3: Dense view synthesis
│   │   └── gaussian.py         # Stage 4: 3D-GS export
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── camera.py           # Camera utilities
│       ├── render.py           # Rendering tools
│       └── io.py               # File I/O
├── configs/                    # Configuration files
│   └── default.yaml
├── extern/                     # External dependencies (clone manually)
│   ├── MVDiffusion/
│   ├── ViewCrafter/
│   └── dust3r/
├── checkpoints/                # Model weights (download manually)
│   ├── mvdiffusion/
│   │   └── pano.ckpt
│   ├── viewcrafter/
│   │   └── model_sparse.ckpt
│   └── dust3r/
│       └── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
├── output/                     # Output directory
├── requirements.txt
└── README.md
```

## 📤 Output Structure

Results are saved to `output/<timestamp>/`:

```
output/20260101_120000/
├── PIPELINE_SUMMARY.txt        # Complete pipeline summary (NEW)
├── stage1_mvdiffusion/         # Stage 1 output
├── stage2_pointcloud/          # Stage 2 output
├── stage3_viewcrafter/         # Stage 3 output
├── stage4_gaussian/            # Stage 4 metadata (NEW)
└── 3dgs/                       # Stage 4 output (COLMAP format)
    ├── images/                 # Training images
    ├── sparse/0/               # COLMAP sparse reconstruction
    ├── training_logs/          # Training logs (if training enabled)
    └── output/                 # Trained model (if training enabled)
```

For detailed output structure, see [OUTPUT_STRUCTURE.md](./OUTPUT_STRUCTURE.md).

## 📚 Citation

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
