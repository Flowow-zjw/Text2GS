"""
Main Text2GS Pipeline
"""

import os
import json
import numpy as np
import torch
from typing import Any, Dict, Optional
from datetime import datetime

from .stages import (
    MVDiffusionStage,
    PointCloudStage,
    ViewCrafterStage,
    GaussianStage,
)
from .utils.io import save_image, save_video, save_pointcloud


class Text2GSPipeline:
    """
    Complete Text-to-3DGS Pipeline
    
    Stages:
        1. MVDiffusion: Text -> 8 panoramic views
        2. PointCloud: Views -> Point cloud + poses
        3. ViewCrafter: Sparse views -> Dense views
        4. Gaussian: Export/Train 3D-GS
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get("device", "cuda:0")
        self.output_dir = config.get("output_dir", "./output")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize stages (lazy loading)
        self.stages = {
            "mvdiffusion": None,
            "pointcloud": None,
            "viewcrafter": None,
            "gaussian": None,
        }
        
        self._loaded_stages = set()
    
    def _get_stage(self, name: str):
        """Get or create a stage"""
        if self.stages[name] is None:
            stage_config = self.config.get(name, {})
            stage_config.update(self.config.get("paths", {}))
            
            if name == "mvdiffusion":
                self.stages[name] = MVDiffusionStage(stage_config, self.device)
            elif name == "pointcloud":
                self.stages[name] = PointCloudStage(stage_config, self.device)
            elif name == "viewcrafter":
                self.stages[name] = ViewCrafterStage(stage_config, self.device)
            elif name == "gaussian":
                self.stages[name] = GaussianStage(stage_config, self.device)
        
        if name not in self._loaded_stages:
            print(f"Loading {name} model...")
            self.stages[name].load_model()
            self._loaded_stages.add(name)
        
        return self.stages[name]
    
    def _unload_stage(self, name: str):
        """Unload a stage to free memory"""
        if name in self._loaded_stages:
            print(f"  Unloading {name} stage...")
            self.stages[name].unload_model()
            self._loaded_stages.discard(name)
            torch.cuda.empty_cache()
            print(f"  {name} stage unloaded successfully")
        else:
            print(f"  Warning: {name} not in loaded stages, skipping unload")
    
    def run(self, text: str, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            text: Text prompt for generation
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print(f"Text2GS Pipeline")
        print(f"Prompt: {text[:50]}...")
        print(f"Output: {self.run_dir}")
        print("=" * 60)
        
        results = {"prompt": text, "output_dir": self.run_dir}
        
        # Stage 1: MVDiffusion
        print("\n[Stage 1/4] MVDiffusion - Generating panoramic views...")
        stage1 = self._get_stage("mvdiffusion")
        stage1_out = stage1.run({"text": text})
        results["stage1"] = stage1_out
        
        if save_intermediate:
            self._save_stage1(stage1_out)
        
        # Unload Stage 1 model before Stage 2 to free memory
        if self.config.get("unload_between_stages", False):
            self._unload_stage("mvdiffusion")
            torch.cuda.empty_cache()
            print("  Unloaded Stage 1 model to free memory")
        
        # Stage 2: Point Cloud
        print("\n[Stage 2/4] DUSt3R - Reconstructing point cloud...")
        stage2 = self._get_stage("pointcloud")
        stage2_out = stage2.run({
            "images": stage1_out["images"],
            "temp_dir": os.path.join(self.run_dir, "temp"),
        })
        results["stage2"] = stage2_out
        
        if save_intermediate:
            self._save_stage2(stage2_out)
        
        # Unload Stage 2 model before Stage 3 to free memory
        if self.config.get("unload_between_stages", False):
            self._unload_stage("pointcloud")
            import torch
            torch.cuda.empty_cache()
            print("  Unloaded Stage 2 model to free memory")
        
        # Stage 3: ViewCrafter
        print("\n[Stage 3/4] ViewCrafter - Generating dense views...")
        stage3 = self._get_stage("viewcrafter")
        # Pass original high-res images from MVDiffusion
        stage2_out["original_images"] = stage1_out["images"]
        stage3_out = stage3.run(stage2_out)
        results["stage3"] = stage3_out
        
        if save_intermediate:
            self._save_stage3(stage3_out)
        
        if self.config.get("unload_between_stages", False):
            self._unload_stage("viewcrafter")
        
        # Stage 4: 3D-GS
        print("\n[Stage 4/4] 3D-GS - Exporting data...")
        stage4 = self._get_stage("gaussian")
        stage4_out = stage4.run({
            **stage3_out,
            "output_dir": os.path.join(self.run_dir, "3dgs"),
        })
        results["stage4"] = stage4_out
        
        # Save Stage 4 metadata
        if save_intermediate:
            self._save_stage4(stage4_out)
        
        # Save pipeline summary
        self._save_pipeline_summary(results)
        
        # Compress results if requested
        if self.config.get("compress_results", False):
            self._compress_results()
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.run_dir}")
        print("=" * 60)
        
        return results
    
    def _compress_results(self) -> None:
        """Compress pipeline results"""
        import tarfile
        
        compress_mode = self.config.get("compress_mode", "minimal")
        
        print(f"\n[Compression] Compressing results ({compress_mode} mode)...")
        
        archive_name = f"{os.path.basename(self.run_dir)}_{compress_mode}.tar.gz"
        archive_path = os.path.join(os.path.dirname(self.run_dir), archive_name)
        
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                if compress_mode == "minimal":
                    # 关键文件：训练模型 + Stage 1 + Stage 2 + 总结
                    patterns = [
                        "PIPELINE_SUMMARY.txt",
                        "stage1_mvdiffusion",
                        "stage2_pointcloud",
                        "stage4_gaussian",
                        "3dgs/output",
                        "3dgs/training_logs/training_status.txt",
                        "3dgs/metadata.json",
                    ]
                elif compress_mode == "model":
                    # 仅模型
                    patterns = ["3dgs/output"]
                else:  # full
                    # 完整目录
                    tar.add(self.run_dir, arcname=os.path.basename(self.run_dir))
                    print(f"  ✓ Compressed to: {archive_path}")
                    return
                
                # 添加指定文件/目录
                for pattern in patterns:
                    path = os.path.join(self.run_dir, pattern)
                    if os.path.exists(path):
                        arcname = os.path.join(os.path.basename(self.run_dir), pattern)
                        tar.add(path, arcname=arcname)
                        print(f"  ✓ Added: {pattern}")
            
            # 显示压缩信息
            size = os.path.getsize(archive_path) / (1024 * 1024)
            print(f"  ✓ Compressed to: {archive_path} ({size:.1f} MB)")
            
        except Exception as e:
            print(f"  ✗ Compression failed: {e}")
    
    def _save_stage1(self, data: Dict[str, Any]) -> None:
        """Save Stage 1 outputs: images, cameras, prompt"""
        stage_dir = os.path.join(self.run_dir, "stage1_mvdiffusion")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save images
        images = data["images"]
        for i, img in enumerate(images):
            save_image(img, os.path.join(stage_dir, f"view_{i:02d}.png"))
        
        # Save prompt
        with open(os.path.join(stage_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(data["prompt"])
        
        # Save camera parameters
        cameras = data["cameras"]
        np.savez(
            os.path.join(stage_dir, "cameras.npz"),
            K=cameras["K"],
            R=cameras["R"],
            resolution=cameras["resolution"],
            fov=cameras["fov"]
        )
        
        # Save metadata
        metadata = {
            "num_views": len(images),
            "resolution": int(cameras["resolution"]),
            "fov": int(cameras["fov"]),
            "prompt": data["prompt"]
        }
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {len(images)} images to {stage_dir}")
    
    def _save_stage2(self, data: Dict[str, Any]) -> None:
        """Save Stage 2 outputs: point cloud, cameras, depths, images"""
        from dust3r.utils.device import to_numpy
        
        stage_dir = os.path.join(self.run_dir, "stage2_pointcloud")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save point cloud
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
        
        # Save images
        images_dir = os.path.join(stage_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(images_dir, f"view_{i:02d}.png"))
        
        # Save depth maps
        depths_dir = os.path.join(stage_dir, "depths")
        os.makedirs(depths_dir, exist_ok=True)
        depths = to_numpy(data["depths"])
        for i, depth in enumerate(depths):
            # Normalize depth for visualization
            d_min, d_max = depth.min(), depth.max()
            depth_vis = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
            save_image(depth_vis, os.path.join(depths_dir, f"depth_{i:02d}.png"))
            # Also save raw depth
            np.save(os.path.join(depths_dir, f"depth_{i:02d}.npy"), depth)
        
        # Save camera poses
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
        
        # Save metadata
        metadata = {
            "num_views": len(imgs),
            "num_points": len(pts),
            "image_shape": data["image_shape"].tolist() if hasattr(data["image_shape"], "tolist") else list(data["image_shape"])
        }
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved point cloud ({len(pts)} points) to {stage_dir}")
    
    def _save_stage3(self, data: Dict[str, Any]) -> None:
        """Save Stage 3 outputs: videos, reconstructed point cloud, all frames"""
        from dust3r.utils.device import to_numpy
        
        stage_dir = os.path.join(self.run_dir, "stage3_viewcrafter")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save video
        videos_dir = os.path.join(stage_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        all_views = data["all_views"][0]  # Concatenated tensor
        frames = (all_views + 1) / 2  # [-1,1] -> [0,1]
        save_video(frames, os.path.join(videos_dir, "generated_views.mp4"))
        
        # Save sampled frames used for reconstruction (not all 169 frames to save disk space)
        sampled_indices = data.get("sampled_indices", [])
        if sampled_indices:
            frames_dir = os.path.join(stage_dir, "sampled_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for idx in sampled_indices:
                frame = ((all_views[idx].numpy() + 1) / 2 * 255).astype(np.uint8)
                save_image(frame, os.path.join(frames_dir, f"frame_{idx:03d}.png"))
            print(f"  Saved {len(sampled_indices)} sampled frames (instead of all {all_views.shape[0]} frames to save space)")
        
        # Save reconstructed point cloud (from DUSt3R)
        pts3d = to_numpy(data["pts3d"])
        
        imgs_raw = data["images"]
        if isinstance(imgs_raw, list):
            imgs = [to_numpy(img) if not isinstance(img, np.ndarray) else img for img in imgs_raw]
        else:
            imgs = imgs_raw if isinstance(imgs_raw, np.ndarray) else to_numpy(imgs_raw)
        
        masks = data.get("masks")
        
        if masks is not None:
            # Convert masks to numpy if they are tensors
            if isinstance(masks, list):
                masks = [to_numpy(m) if not isinstance(m, np.ndarray) else m for m in masks]
            else:
                masks = to_numpy(masks) if not isinstance(masks, np.ndarray) else masks
            
            pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)])
            cols = np.concatenate([p[m] for p, m in zip(imgs, masks)])
        else:
            pts = np.concatenate([p.reshape(-1, 3) for p in pts3d])
            cols = np.concatenate([p.reshape(-1, 3) for p in imgs])
        
        save_pointcloud(pts, cols, os.path.join(stage_dir, "pointcloud_reconstructed.ply"))
        
        # Save reconstructed images used for training
        reconstructed_images_dir = os.path.join(stage_dir, "reconstructed_images")
        os.makedirs(reconstructed_images_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(reconstructed_images_dir, f"image_{i:04d}.png"))
        print(f"  Saved {len(imgs)} reconstructed images to {reconstructed_images_dir}")
        
        # Save camera poses (reconstructed from DUSt3R)
        c2ws = data["c2ws"]
        if hasattr(c2ws, 'cpu'):
            c2ws = c2ws.cpu().numpy()
        
        focals = data["focals"]
        if hasattr(focals, 'cpu'):
            focals = focals.cpu().numpy()
            
        principal_points = data["principal_points"]
        if hasattr(principal_points, 'cpu'):
            principal_points = principal_points.cpu().numpy()
        
        np.savez(
            os.path.join(stage_dir, "cameras.npz"),
            c2ws=c2ws,
            focals=focals,
            principal_points=principal_points
        )
        
        # Save metadata
        num_input_views = data.get("num_input_views", 8)
        video_length = data.get("video_length", 25)
        num_sampled = data.get("num_sampled_frames", 0)
        sampled_indices = data.get("sampled_indices", [])
        original_indices = data.get("original_frame_indices", [])
        total_frames = all_views.shape[0]
        
        metadata = {
            "num_input_views": num_input_views,
            "video_length": video_length,
            "total_generated_frames": total_frames,
            "num_sampled_frames": num_sampled,
            "num_reconstructed_views": len(imgs),
            "num_points": len(pts),
            "sampled_indices": sampled_indices,
            "original_frame_indices": original_indices,
            "workflow": "ViewCrafter interpolation -> Remove duplicates -> Uniform sampling (保留原始帧) -> DUSt3R reconstruction"
        }
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {total_frames} total frames, {len(sampled_indices)} sampled frames, {len(imgs)} reconstructed images to {stage_dir}")
    
    def _save_stage4(self, data: Dict[str, Any]) -> None:
        """Save Stage 4 outputs: metadata and training info"""
        stage_dir = os.path.join(self.run_dir, "stage4_gaussian")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "export_dir": data.get("export_dir"),
            "colmap_dir": data.get("colmap_dir"),
            "images_dir": data.get("images_dir"),
            "num_images": data.get("num_images"),
            "num_points": data.get("num_points"),
            "training_enabled": "training" in data,
        }
        
        # Add training info if available
        if "training" in data:
            training_info = data["training"]
            metadata["training"] = {
                "model_path": training_info.get("model_path"),
                "iterations": training_info.get("iterations"),
                "success": training_info.get("success"),
                "log_file": training_info.get("log_file"),
                "config_file": training_info.get("config_file"),
                "status_file": training_info.get("status_file"),
            }
            metadata["trained_model_path"] = data.get("trained_model_path")
        
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create a summary text file
        with open(os.path.join(stage_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("Stage 4: 3D Gaussian Splatting\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Export Directory: {data.get('export_dir')}\n")
            f.write(f"COLMAP Directory: {data.get('colmap_dir')}\n")
            f.write(f"Images Directory: {data.get('images_dir')}\n")
            f.write(f"Number of Images: {data.get('num_images')}\n")
            f.write(f"Number of Points: {data.get('num_points')}\n\n")
            
            if "training" in data:
                f.write("Training Information:\n")
                f.write("-" * 60 + "\n")
                training_info = data["training"]
                f.write(f"Model Path: {training_info.get('model_path')}\n")
                f.write(f"Iterations: {training_info.get('iterations')}\n")
                f.write(f"Success: {training_info.get('success')}\n")
                f.write(f"Log File: {training_info.get('log_file')}\n")
                f.write(f"Config File: {training_info.get('config_file')}\n")
                f.write(f"Status File: {training_info.get('status_file')}\n")
                f.write(f"\nTrained Model: {data.get('trained_model_path')}\n")
            else:
                f.write("Training: Not performed (export only)\n")
        
        print(f"  Saved Stage 4 metadata to {stage_dir}")
    
    def _save_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Save complete pipeline summary"""
        summary_file = os.path.join(self.run_dir, "PIPELINE_SUMMARY.txt")
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("TEXT2GS PIPELINE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic info
            f.write(f"Prompt: {results['prompt']}\n")
            f.write(f"Output Directory: {results['output_dir']}\n")
            f.write(f"Timestamp: {os.path.basename(self.run_dir)}\n\n")
            
            # Stage 1
            f.write("-" * 80 + "\n")
            f.write("STAGE 1: MVDiffusion - Multi-view Generation\n")
            f.write("-" * 80 + "\n")
            stage1 = results.get("stage1", {})
            f.write(f"Number of Views: {len(stage1.get('images', []))}\n")
            cameras = stage1.get("cameras", {})
            f.write(f"Resolution: {cameras.get('resolution', 'N/A')}\n")
            f.write(f"FOV: {cameras.get('fov', 'N/A')}°\n")
            f.write(f"Output: stage1_mvdiffusion/\n\n")
            
            # Stage 2
            f.write("-" * 80 + "\n")
            f.write("STAGE 2: DUSt3R - Point Cloud Reconstruction\n")
            f.write("-" * 80 + "\n")
            stage2 = results.get("stage2", {})
            pts3d = stage2.get("pts3d", [])
            if pts3d:
                total_points = sum(p.numel() // 3 for p in pts3d)
                f.write(f"Total Points: {total_points:,}\n")
            f.write(f"Number of Views: {len(stage2.get('images', []))}\n")
            f.write(f"Output: stage2_pointcloud/\n")
            f.write(f"  - pointcloud.ply\n")
            f.write(f"  - images/\n")
            f.write(f"  - depths/\n")
            f.write(f"  - cameras.npz\n\n")
            
            # Stage 3
            f.write("-" * 80 + "\n")
            f.write("STAGE 3: ViewCrafter - Dense View Synthesis\n")
            f.write("-" * 80 + "\n")
            stage3 = results.get("stage3", {})
            all_views = stage3.get("all_views", [])
            if all_views and len(all_views) > 0:
                num_frames = all_views[0].shape[0]
                f.write(f"Total Frames Generated: {num_frames}\n")
            f.write(f"Input Views: {stage3.get('num_input_views', 'N/A')}\n")
            f.write(f"Video Length per Clip: {stage3.get('video_length', 'N/A')}\n")
            f.write(f"Output: stage3_viewcrafter/\n")
            f.write(f"  - videos/\n")
            f.write(f"  - frames/\n")
            f.write(f"  - pointcloud.ply\n")
            f.write(f"  - cameras.npz\n\n")
            
            # Stage 4
            f.write("-" * 80 + "\n")
            f.write("STAGE 4: 3D Gaussian Splatting\n")
            f.write("-" * 80 + "\n")
            stage4 = results.get("stage4", {})
            f.write(f"Export Directory: {stage4.get('export_dir', 'N/A')}\n")
            f.write(f"COLMAP Directory: {stage4.get('colmap_dir', 'N/A')}\n")
            f.write(f"Number of Images: {stage4.get('num_images', 'N/A')}\n")
            f.write(f"Number of Points: {stage4.get('num_points', 'N/A')}\n")
            
            if "training" in stage4:
                f.write(f"\nTraining Status: COMPLETED\n")
                training = stage4["training"]
                f.write(f"Iterations: {training.get('iterations', 'N/A')}\n")
                f.write(f"Model Path: {training.get('model_path', 'N/A')}\n")
                f.write(f"Success: {training.get('success', False)}\n")
            else:
                f.write(f"\nTraining Status: NOT PERFORMED (export only)\n")
            
            f.write(f"\nOutput: 3dgs/\n")
            f.write(f"  - images/\n")
            f.write(f"  - sparse/0/\n")
            if "training" in stage4:
                f.write(f"  - output/ (trained model)\n")
            f.write("\n")
            
            # Summary
            f.write("=" * 80 + "\n")
            f.write("DIRECTORY STRUCTURE\n")
            f.write("=" * 80 + "\n")
            f.write(f"{self.run_dir}/\n")
            f.write(f"├── PIPELINE_SUMMARY.txt (this file)\n")
            f.write(f"├── stage1_mvdiffusion/\n")
            f.write(f"│   ├── view_00.png ~ view_07.png\n")
            f.write(f"│   ├── cameras.npz\n")
            f.write(f"│   ├── prompt.txt\n")
            f.write(f"│   └── metadata.json\n")
            f.write(f"├── stage2_pointcloud/\n")
            f.write(f"│   ├── pointcloud.ply\n")
            f.write(f"│   ├── images/\n")
            f.write(f"│   ├── depths/\n")
            f.write(f"│   ├── cameras.npz\n")
            f.write(f"│   └── metadata.json\n")
            f.write(f"├── stage3_viewcrafter/\n")
            f.write(f"│   ├── videos/\n")
            f.write(f"│   ├── frames/\n")
            f.write(f"│   ├── pointcloud.ply\n")
            f.write(f"│   ├── cameras.npz\n")
            f.write(f"│   └── metadata.json\n")
            f.write(f"├── stage4_gaussian/\n")
            f.write(f"│   ├── metadata.json\n")
            f.write(f"│   └── summary.txt\n")
            f.write(f"└── 3dgs/\n")
            f.write(f"    ├── images/\n")
            f.write(f"    ├── sparse/0/\n")
            f.write(f"    │   ├── cameras.txt\n")
            f.write(f"    │   ├── images.txt\n")
            f.write(f"    │   └── points3D.txt\n")
            if "training" in stage4:
                f.write(f"    ├── output/ (trained model)\n")
                f.write(f"    │   ├── point_cloud/\n")
                f.write(f"    │   ├── cameras.json\n")
                f.write(f"    │   └── cfg_args\n")
            f.write(f"    └── metadata.json\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("=" * 80 + "\n")
            if "training" in stage4:
                f.write("View the trained model:\n")
                f.write(f"  cd /root/autodl-tmp/gaussian-splatting\n")
                f.write(f"  python viewer.py -m {stage4['training']['model_path']}\n")
            else:
                f.write("Train the model manually:\n")
                f.write(f"  cd /root/autodl-tmp/gaussian-splatting\n")
                f.write(f"  python train.py -s {stage4.get('export_dir', 'N/A')} --iterations 30000\n")
            f.write("\n")
        
        print(f"\n  Saved pipeline summary to {summary_file}")
