#!/usr/bin/env python
"""
从 Stage 3 输出直接训练 3D-GS
一步到位：转换 + 训练

用法:
    python train_from_stage3.py --stage3-dir output/TIMESTAMP/stage3_viewcrafter --iterations 7000
"""

import os
import sys
import argparse
import subprocess
import yaml
from convert_stage3_to_colmap import convert_to_colmap


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 3D-GS directly from Stage 3 output"
    )
    
    parser.add_argument(
        "--stage3-dir",
        type=str,
        required=True,
        help="Path to stage3_viewcrafter directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: stage3_dir/../3dgs_direct)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to config YAML file (default: ./configs/default.yaml)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations (overrides config, default from config or 7000)"
    )
    
    parser.add_argument(
        "--gs-path",
        type=str,
        default=None,
        help="Path to gaussian-splatting installation (overrides config)"
    )
    
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum initial points (overrides config)"
    )
    
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert to COLMAP format, don't train"
    )
    
    # Advanced training parameters
    parser.add_argument("--sh-degree", type=int, default=None, help="Spherical harmonics degree (overrides config)")
    parser.add_argument("--lambda-dssim", type=float, default=None, help="SSIM loss weight (overrides config)")
    parser.add_argument("--opacity-reset-interval", type=int, default=None, help="Opacity reset interval (overrides config)")
    parser.add_argument("--densify-grad-threshold", type=float, default=None, help="Densification gradient threshold (overrides config)")
    
    return parser.parse_args()


def train_3dgs(data_dir: str, gs_path: str, iterations: int, test_iterations: list = None, 
               save_iterations: list = None, checkpoint_iterations: list = None, **kwargs):
    """Train 3D Gaussian Splatting"""
    print("\n" + "=" * 60)
    print("Training 3D Gaussian Splatting")
    print("=" * 60)
    
    if not os.path.exists(gs_path):
        print(f"✗ Error: gaussian-splatting not found at {gs_path}")
        return False
    
    train_script = os.path.join(gs_path, "train.py")
    output_model_dir = os.path.join(data_dir, "output")
    
    cmd = [
        "python", train_script,
        "-s", data_dir,
        "-m", output_model_dir,
        "--iterations", str(iterations),
        "--eval"
    ]
    
    # Add test iterations
    if test_iterations:
        for it in test_iterations:
            cmd.extend(["--test_iterations", str(it)])
    
    # Add save iterations
    if save_iterations:
        for it in save_iterations:
            cmd.extend(["--save_iterations", str(it)])
    
    # Add checkpoint iterations
    if checkpoint_iterations:
        for it in checkpoint_iterations:
            cmd.extend(["--checkpoint_iterations", str(it)])
    
    # Add optional parameters
    if kwargs.get("sh_degree") is not None:
        cmd.extend(["--sh_degree", str(kwargs["sh_degree"])])
    if kwargs.get("lambda_dssim") is not None:
        cmd.extend(["--lambda_dssim", str(kwargs["lambda_dssim"])])
    if kwargs.get("opacity_reset_interval") is not None:
        cmd.extend(["--opacity_reset_interval", str(kwargs["opacity_reset_interval"])])
    if kwargs.get("densify_grad_threshold") is not None:
        cmd.extend(["--densify_grad_threshold", str(kwargs["densify_grad_threshold"])])
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Working directory: {gs_path}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=gs_path,
            check=True,
            text=True
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"\nTrained model: {output_model_dir}")
        print("\nTo view the model:")
        print(f"  cd {gs_path}")
        print(f"  python viewer.py -m {output_model_dir}")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    args = parse_args()
    
    # Load config file
    config = load_config(args.config)
    gaussian_config = config.get("gaussian", {})
    paths_config = config.get("paths", {})
    
    # Get parameters from config with CLI overrides
    iterations = args.iterations if args.iterations is not None else gaussian_config.get("iterations", 7000)
    gs_path = args.gs_path if args.gs_path is not None else paths_config.get("gaussian_splatting_path", "/root/autodl-tmp/gaussian-splatting")
    max_points = args.max_points if args.max_points is not None else gaussian_config.get("max_init_points", 500000)
    
    # Advanced training parameters from config
    sh_degree = args.sh_degree if args.sh_degree is not None else gaussian_config.get("sh_degree")
    lambda_dssim = args.lambda_dssim if args.lambda_dssim is not None else gaussian_config.get("lambda_dssim")
    opacity_reset_interval = args.opacity_reset_interval if args.opacity_reset_interval is not None else gaussian_config.get("opacity_reset_interval")
    densify_grad_threshold = args.densify_grad_threshold if args.densify_grad_threshold is not None else gaussian_config.get("densify_grad_threshold")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        parent_dir = os.path.dirname(args.stage3_dir)
        output_dir = os.path.join(parent_dir, "3dgs_direct")
    
    print("=" * 60)
    print("Train 3D-GS from Stage 3 Output")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Stage 3 directory: {args.stage3_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Gaussian-splatting path: {gs_path}")
    print(f"Iterations: {iterations}")
    print(f"Max initial points: {max_points}")
    if sh_degree is not None:
        print(f"SH degree: {sh_degree}")
    if lambda_dssim is not None:
        print(f"Lambda DSSIM: {lambda_dssim}")
    if opacity_reset_interval is not None:
        print(f"Opacity reset interval: {opacity_reset_interval}")
    if densify_grad_threshold is not None:
        print(f"Densify grad threshold: {densify_grad_threshold}")
    print("=" * 60)
    print()
    
    # Step 1: Convert to COLMAP format
    try:
        print("[Step 1/2] Converting to COLMAP format...")
        metadata = convert_to_colmap(args.stage3_dir, output_dir, max_points)
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if args.convert_only:
        print("\n✓ Conversion complete (training skipped)")
        return 0
    
    # Step 2: Train 3D-GS
    print("\n[Step 2/2] Training 3D Gaussian Splatting...")
    
    # Build training kwargs
    training_kwargs = {}
    if sh_degree is not None:
        training_kwargs["sh_degree"] = sh_degree
    if lambda_dssim is not None:
        training_kwargs["lambda_dssim"] = lambda_dssim
    if opacity_reset_interval is not None:
        training_kwargs["opacity_reset_interval"] = opacity_reset_interval
    if densify_grad_threshold is not None:
        training_kwargs["densify_grad_threshold"] = densify_grad_threshold
    
    # Get test/save/checkpoint iterations from config (not exposed as CLI args for simplicity)
    test_iterations = [iterations]  # Test at final iteration
    save_iterations = [iterations]  # Save at final iteration
    checkpoint_iterations = []  # No intermediate checkpoints by default
    
    success = train_3dgs(
        output_dir,
        gs_path,
        iterations,
        test_iterations=test_iterations,
        save_iterations=save_iterations,
        checkpoint_iterations=checkpoint_iterations,
        **training_kwargs
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
