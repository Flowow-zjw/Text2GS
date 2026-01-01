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
            self.stages[name].unload_model()
            self._loaded_stages.discard(name)
            torch.cuda.empty_cache()
    
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
        
        # Optionally unload to save memory
        if self.config.get("unload_between_stages", False):
            self._unload_stage("mvdiffusion")
        
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
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.run_dir}")
        print("=" * 60)
        
        return results
    
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
        """Save Stage 3 outputs: videos, point cloud, all frames"""
        from dust3r.utils.device import to_numpy
        
        stage_dir = os.path.join(self.run_dir, "stage3_viewcrafter")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save video
        videos_dir = os.path.join(stage_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        all_views = data["all_views"][0]  # Now it's a single concatenated tensor
        frames = (all_views + 1) / 2  # [-1,1] -> [0,1]
        save_video(frames, os.path.join(videos_dir, "generated_views.mp4"))
        
        # Save all generated frames as images
        frames_dir = os.path.join(stage_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for j in range(all_views.shape[0]):
            frame = ((all_views[j].numpy() + 1) / 2 * 255).astype(np.uint8)
            save_image(frame, os.path.join(frames_dir, f"frame_{j:03d}.png"))
        
        # Save point cloud
        pts3d = to_numpy(data["pts3d"])
        
        imgs_raw = data["images"]
        if isinstance(imgs_raw, list):
            imgs = [to_numpy(img) if not isinstance(img, np.ndarray) else img for img in imgs_raw]
        else:
            imgs = imgs_raw if isinstance(imgs_raw, np.ndarray) else to_numpy(imgs_raw)
        
        masks = data.get("masks")
        
        if masks is not None:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)])
            cols = np.concatenate([p[m] for p, m in zip(imgs, masks)])
        else:
            pts = np.concatenate([p.reshape(-1, 3) for p in pts3d])
            cols = np.concatenate([p.reshape(-1, 3) for p in imgs])
        
        save_pointcloud(pts, cols, os.path.join(stage_dir, "pointcloud.ply"))
        
        # Save camera poses (original + interpolated)
        c2ws = data["c2ws"]
        if hasattr(c2ws, 'cpu'):
            c2ws = c2ws.cpu().numpy()
        
        c2ws_interp = data.get("c2ws_interp")
        if c2ws_interp is not None and hasattr(c2ws_interp, 'cpu'):
            c2ws_interp = c2ws_interp.cpu().numpy()
        
        focals = data["focals"]
        if hasattr(focals, 'cpu'):
            focals = focals.cpu().numpy()
            
        principal_points = data["principal_points"]
        if hasattr(principal_points, 'cpu'):
            principal_points = principal_points.cpu().numpy()
        
        np.savez(
            os.path.join(stage_dir, "cameras.npz"),
            c2ws=c2ws,
            c2ws_interp=c2ws_interp,
            focals=focals,
            principal_points=principal_points
        )
        
        # Save metadata
        num_input_views = data.get("num_input_views", len(imgs))
        video_length = data.get("video_length", 25)
        total_frames = all_views.shape[0]
        
        metadata = {
            "num_input_views": num_input_views,
            "video_length": video_length,
            "total_frames": total_frames,
            "num_points": len(pts),
            "frame_interval_degrees": 360.0 / total_frames if total_frames > 0 else 0
        }
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {total_frames} frames (interval: {metadata['frame_interval_degrees']:.1f}°) to {stage_dir}")
