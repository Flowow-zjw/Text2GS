"""
Main entry point for Text2GS
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any

from .pipeline import Text2GSPipeline


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge two config dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Text2GS: Text to 3D Gaussian Splatting")
    
    # Required
    parser.add_argument("--text", type=str, required=True,
                        help="Text prompt for generation")
    
    # Config
    parser.add_argument("--config", type=str, default="./configs/default.yaml",
                        help="Path to config YAML file")
    
    # Output
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    # Paths (override config)
    parser.add_argument("--mvdiffusion-path", type=str, default=None,
                        help="Path to MVDiffusion")
    parser.add_argument("--viewcrafter-path", type=str, default=None,
                        help="Path to ViewCrafter")
    parser.add_argument("--dust3r-path", type=str, default=None,
                        help="Path to DUSt3R")
    
    # Checkpoints
    parser.add_argument("--mvdiffusion-ckpt", type=str, default=None,
                        help="MVDiffusion checkpoint")
    parser.add_argument("--viewcrafter-ckpt", type=str, default=None,
                        help="ViewCrafter checkpoint")
    parser.add_argument("--dust3r-ckpt", type=str, default=None,
                        help="DUSt3R checkpoint")
    
    # Options
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="ViewCrafter iterations")
    parser.add_argument("--no-save-intermediate", action="store_true",
                        help="Don't save intermediate results")
    parser.add_argument("--unload-between-stages", action="store_true",
                        help="Unload models between stages to save memory")
    
    # Compression
    parser.add_argument("--compress", action="store_true",
                        help="Compress results after completion")
    parser.add_argument("--compress-mode", type=str, default="minimal",
                        choices=["minimal", "model", "full"],
                        help="Compression mode: minimal (default), model, or full")
    
    # 3D-GS Training
    parser.add_argument("--train-3dgs", action="store_true",
                        help="Train 3D Gaussian Splatting after export")
    parser.add_argument("--gs-iterations", type=int, default=None,
                        help="3D-GS training iterations (overrides config)")
    parser.add_argument("--gs-path", type=str, default=None,
                        help="Path to gaussian-splatting installation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load base config
    if args.config:
        config = load_config(args.config)
    else:
        # Default config
        config = {
            "device": "cuda:0",
            "output_dir": "./output",
            "paths": {
                "mvdiffusion_path": "./extern/MVDiffusion",
                "viewcrafter_path": "./extern/ViewCrafter",
                "dust3r_path": "./extern/dust3r",
            },
            "mvdiffusion": {
                "num_views": 8,
                "resolution": 512,
                "fov": 90,
                "guidance_scale": 9.0,
                "diff_timesteps": 50,
                "checkpoint": "./checkpoints/mvdiffusion/pano.ckpt",
            },
            "pointcloud": {
                "batch_size": 1,
                "niter": 300,
                "lr": 0.01,
                "checkpoint": "./checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
            },
            "viewcrafter": {
                "num_iterations": 3,
                "video_length": 25,
                "ddim_steps": 50,
                "guidance_scale": 7.5,
                # Use sparse model for multi-view input
                "checkpoint": "./checkpoints/viewcrafter/model_sparse.ckpt",
                "bg_trd": 0.2,
            },
            "gaussian": {
                "iterations": 30000,
                "export_only": False,
                "test_iterations": [7000, 30000],
                "save_iterations": [7000, 30000],
                "checkpoint_iterations": [],
                "gaussian_splatting_path": "/root/autodl-tmp/gaussian-splatting",
            },
        }
    
    # Override with command line args
    config["device"] = args.device
    config["output_dir"] = args.output
    
    if args.mvdiffusion_path:
        config["paths"]["mvdiffusion_path"] = args.mvdiffusion_path
    if args.viewcrafter_path:
        config["paths"]["viewcrafter_path"] = args.viewcrafter_path
    if args.dust3r_path:
        config["paths"]["dust3r_path"] = args.dust3r_path
    
    if args.mvdiffusion_ckpt:
        config["mvdiffusion"]["checkpoint"] = args.mvdiffusion_ckpt
    if args.viewcrafter_ckpt:
        config["viewcrafter"]["checkpoint"] = args.viewcrafter_ckpt
    if args.dust3r_ckpt:
        config["pointcloud"]["checkpoint"] = args.dust3r_ckpt
    
    if args.num_iterations:
        config["viewcrafter"]["num_iterations"] = args.num_iterations
    
    config["unload_between_stages"] = args.unload_between_stages
    
    # Compression options
    config["compress_results"] = args.compress
    config["compress_mode"] = args.compress_mode
    
    # 3D-GS training options
    if args.train_3dgs:
        config["gaussian"]["export_only"] = False
    if args.gs_iterations:
        config["gaussian"]["iterations"] = args.gs_iterations
    if args.gs_path:
        config["paths"]["gaussian_splatting_path"] = args.gs_path
        config["gaussian"]["gaussian_splatting_path"] = args.gs_path
    
    # Create and run pipeline
    pipeline = Text2GSPipeline(config)
    results = pipeline.run(
        text=args.text,
        save_intermediate=not args.no_save_intermediate
    )
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
