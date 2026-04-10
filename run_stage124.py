#!/usr/bin/env python
"""
运行 Text2GS Stage 1, 2, 4（跳过 Stage 3）
适用于快速测试或不需要密集视图的场景
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

from text2gs.stages import MVDiffusionStage, PointCloudStage, GaussianStage
from text2gs.utils.io import save_image, save_pointcloud
import numpy as np
import torch
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Text2GS: Stage 1, 2, 4 only")
    
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
    
    # 3D-GS Training
    parser.add_argument("--train-3dgs", action="store_true",
                        help="Train 3D Gaussian Splatting after export")
    parser.add_argument("--gs-iterations", type=int, default=None,
                        help="3D-GS training iterations (overrides config)")
    parser.add_argument("--gs-path", type=str, default=None,
                        help="Path to gaussian-splatting installation")
    
    parser.add_argument("--unload-between-stages", action="store_true",
                        help="Unload models between stages to save memory")
    
    # Compression
    parser.add_argument("--compress", action="store_true",
                        help="Compress results after completion")
    parser.add_argument("--compress-mode", type=str, default="minimal",
                        choices=["minimal", "model", "full"],
                        help="Compression mode: minimal (default), model, or full")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def save_stage1(data, run_dir):
    """Save Stage 1 outputs"""
    stage_dir = os.path.join(run_dir, "stage1_mvdiffusion")
    os.makedirs(stage_dir, exist_ok=True)
    
    images = data["images"]
    for i, img in enumerate(images):
        save_image(img, os.path.join(stage_dir, f"view_{i:02d}.png"))
    
    with open(os.path.join(stage_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(data["prompt"])
    
    cameras = data["cameras"]
    np.savez(
        os.path.join(stage_dir, "cameras.npz"),
        K=cameras["K"],
        R=cameras["R"],
        resolution=cameras["resolution"],
        fov=cameras["fov"]
    )
    
    metadata = {
        "num_views": len(images),
        "resolution": int(cameras["resolution"]),
        "fov": int(cameras["fov"]),
        "prompt": data["prompt"]
    }
    with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved {len(images)} images to {stage_dir}")


def save_stage2(data, run_dir):
    """Save Stage 2 outputs"""
    from dust3r.utils.device import to_numpy
    
    stage_dir = os.path.join(run_dir, "stage2_pointcloud")
    os.makedirs(stage_dir, exist_ok=True)
    
    pts3d = to_numpy(data["pts3d"])
    imgs = to_numpy(data["images"])
    masks = to_numpy(data["masks"]) if data.get("masks") is not None else None
    
    if masks is not None:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)])
        cols = np.concatenate([p[m] for p, m in zip(imgs, masks)])
    else:
        pts = np.concatenate([p.reshape(-1, 3) for p in pts3d])
        cols = np.concatenate([p.reshape(-1, 3) for p in imgs])
    
    save_pointcloud(pts, cols, os.path.join(stage_dir, "pointcloud.ply"))
    
    c2ws = data["c2ws"].cpu().numpy()
    focals = data["focals"].cpu().numpy()
    principal_points = data["principal_points"].cpu().numpy()
    
    np.savez(
        os.path.join(stage_dir, "cameras.npz"),
        c2ws=c2ws,
        focals=focals,
        principal_points=principal_points,
        image_shape=data["image_shape"]
    )
    
    metadata = {
        "num_views": len(imgs),
        "num_points": len(pts),
        "image_shape": data["image_shape"].tolist() if hasattr(data["image_shape"], "tolist") else list(data["image_shape"])
    }
    with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved point cloud ({len(pts)} points) to {stage_dir}")


def save_stage4(data, run_dir):
    """Save Stage 4 metadata"""
    stage_dir = os.path.join(run_dir, "stage4_gaussian")
    os.makedirs(stage_dir, exist_ok=True)
    
    metadata = {
        "export_dir": data.get("export_dir"),
        "colmap_dir": data.get("colmap_dir"),
        "images_dir": data.get("images_dir"),
        "num_images": data.get("num_images"),
        "num_points": data.get("num_points"),
        "training_enabled": "training" in data,
    }
    
    if "training" in data:
        training_info = data["training"]
        metadata["training"] = {
            "model_path": training_info.get("model_path"),
            "iterations": training_info.get("iterations"),
            "success": training_info.get("success"),
            "log_file": training_info.get("log_file"),
        }
    
    with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved Stage 4 metadata to {stage_dir}")


def save_summary(results, run_dir):
    """Save pipeline summary"""
    summary_file = os.path.join(run_dir, "PIPELINE_SUMMARY.txt")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TEXT2GS PIPELINE SUMMARY (Stage 1, 2, 4 only)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Prompt: {results['prompt']}\n")
        f.write(f"Output Directory: {results['output_dir']}\n")
        f.write(f"Note: Stage 3 (ViewCrafter) was skipped\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STAGE 1: MVDiffusion\n")
        f.write("-" * 80 + "\n")
        stage1 = results.get("stage1", {})
        f.write(f"Number of Views: {len(stage1.get('images', []))}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STAGE 2: DUSt3R\n")
        f.write("-" * 80 + "\n")
        stage2 = results.get("stage2", {})
        f.write(f"Number of Views: {len(stage2.get('images', []))}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STAGE 3: ViewCrafter\n")
        f.write("-" * 80 + "\n")
        f.write("SKIPPED\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STAGE 4: 3D Gaussian Splatting\n")
        f.write("-" * 80 + "\n")
        stage4 = results.get("stage4", {})
        f.write(f"Export Directory: {stage4.get('export_dir', 'N/A')}\n")
        f.write(f"Number of Images: {stage4.get('num_images', 'N/A')}\n")
        
        if "training" in stage4:
            f.write(f"\nTraining Status: COMPLETED\n")
            training = stage4["training"]
            f.write(f"Iterations: {training.get('iterations', 'N/A')}\n")
            f.write(f"Model Path: {training.get('model_path', 'N/A')}\n")
        else:
            f.write(f"\nTraining Status: NOT PERFORMED\n")
    
    print(f"\n  Saved pipeline summary to {summary_file}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    config["device"] = args.device
    config["output_dir"] = args.output
    
    # Override paths if provided
    if args.mvdiffusion_path:
        config["paths"]["mvdiffusion_path"] = args.mvdiffusion_path
    if args.viewcrafter_path:
        config["paths"]["viewcrafter_path"] = args.viewcrafter_path
    if args.dust3r_path:
        config["paths"]["dust3r_path"] = args.dust3r_path
    
    if args.gs_path:
        config["paths"]["gaussian_splatting_path"] = args.gs_path
        config["gaussian"]["gaussian_splatting_path"] = args.gs_path
    
    if args.train_3dgs:
        config["gaussian"]["export_only"] = False
    
    if args.gs_iterations:
        config["gaussian"]["iterations"] = args.gs_iterations
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Text2GS Pipeline (Stage 1, 2, 4 only)")
    print(f"Prompt: {args.text[:50]}...")
    print(f"Output: {run_dir}")
    print("=" * 60)
    
    results = {"prompt": args.text, "output_dir": run_dir}
    
    # Stage 1: MVDiffusion
    print("\n[Stage 1/3] MVDiffusion - Generating panoramic views...")
    stage1_config = config.get("mvdiffusion", {})
    stage1_config.update(config.get("paths", {}))
    stage1 = MVDiffusionStage(stage1_config, args.device)
    stage1.load_model()
    stage1_out = stage1.run({"text": args.text})
    results["stage1"] = stage1_out
    save_stage1(stage1_out, run_dir)
    
    if args.unload_between_stages:
        stage1.unload_model()
        torch.cuda.empty_cache()
    
    # Stage 2: Point Cloud
    print("\n[Stage 2/3] DUSt3R - Reconstructing point cloud...")
    stage2_config = config.get("pointcloud", {})
    stage2_config.update(config.get("paths", {}))
    stage2 = PointCloudStage(stage2_config, args.device)
    stage2.load_model()
    stage2_out = stage2.run({
        "images": stage1_out["images"],
        "temp_dir": os.path.join(run_dir, "temp"),
    })
    results["stage2"] = stage2_out
    save_stage2(stage2_out, run_dir)
    
    if args.unload_between_stages:
        stage2.unload_model()
        torch.cuda.empty_cache()
    
    # Stage 3: Skipped
    print("\n[Stage 3/3] ViewCrafter - SKIPPED")
    
    # Stage 4: 3D-GS (using Stage 2 output directly)
    print("\n[Stage 4/3] 3D-GS - Exporting data...")
    stage4_config = config.get("gaussian", {})
    stage4_config.update(config.get("paths", {}))
    stage4 = GaussianStage(stage4_config, args.device)
    stage4.load_model()
    
    # Prepare Stage 4 input from Stage 2 output
    stage4_input = {
        "pts3d": stage2_out["pts3d"],
        "images": stage2_out["images"],
        "c2ws": stage2_out["c2ws"],
        "focals": stage2_out["focals"],
        "principal_points": stage2_out["principal_points"],
        "masks": stage2_out.get("masks"),
        "original_images": stage1_out["images"],  # Use Stage 1 images
        "all_views": [],  # No interpolated views
        "num_input_views": len(stage1_out["images"]),
        "output_dir": os.path.join(run_dir, "3dgs"),
    }
    
    stage4_out = stage4.run(stage4_input)
    results["stage4"] = stage4_out
    save_stage4(stage4_out, run_dir)
    
    # Save summary
    save_summary(results, run_dir)
    
    # Compress results if requested
    if args.compress:
        compress_results(run_dir, args.compress_mode)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)
    
    return results


def compress_results(run_dir, mode="minimal"):
    """Compress pipeline results"""
    import tarfile
    
    print(f"\n[Compression] Compressing results ({mode} mode)...")
    
    archive_name = f"{os.path.basename(run_dir)}_{mode}.tar.gz"
    archive_path = os.path.join(os.path.dirname(run_dir), archive_name)
    
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            if mode == "minimal":
                # 关键文件：训练模型 + Stage 1 + 总结
                patterns = [
                    "PIPELINE_SUMMARY.txt",
                    "stage1_mvdiffusion",
                    "stage4_gaussian",
                    "3dgs/output",
                    "3dgs/training_logs/training_status.txt",
                    "3dgs/metadata.json",
                ]
            elif mode == "model":
                # 仅模型
                patterns = ["3dgs/output"]
            else:  # full
                # 完整目录
                tar.add(run_dir, arcname=os.path.basename(run_dir))
                print(f"  ✓ Compressed to: {archive_path}")
                return archive_path
            
            # 添加指定文件/目录
            for pattern in patterns:
                path = os.path.join(run_dir, pattern)
                if os.path.exists(path):
                    arcname = os.path.join(os.path.basename(run_dir), pattern)
                    tar.add(path, arcname=arcname)
                    print(f"  ✓ Added: {pattern}")
        
        # 显示压缩信息
        size = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"  ✓ Compressed to: {archive_path} ({size:.1f} MB)")
        return archive_path
        
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return None


if __name__ == "__main__":
    main()
