"""
Stage 2: Point Cloud Reconstruction using DUSt3R
"""

import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image

from .base import BaseStage


class PointCloudStage(BaseStage):
    """Reconstruct point cloud from multi-view images using DUSt3R"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        super().__init__(config, device)
        self.batch_size = config.get("batch_size", 1)
        self.niter = config.get("niter", 300)
        self.lr = config.get("lr", 0.01)
        self.schedule = config.get("schedule", "linear")
        self.min_conf_thr = config.get("min_conf_thr", 3.0)
        
    def load_model(self) -> None:
        """Load DUSt3R model"""
        # Add DUSt3R to path
        dust3r_path = self.config.get("dust3r_path", "./extern/dust3r")
        sys.path.insert(0, dust3r_path)
        
        # Try new API first, fall back to old API
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            ckpt_path = self.config.get(
                "checkpoint", 
                "./checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"DUSt3R checkpoint not found: {ckpt_path}")
            
            self.model = AsymmetricCroCo3DStereo.from_pretrained(ckpt_path).to(self.device)
            print(f"Loaded DUSt3R checkpoint from {ckpt_path}")
        except (ImportError, AttributeError):
            # Fall back to old API
            from dust3r.inference import load_model
            ckpt_path = self.config.get(
                "checkpoint", 
                "./checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
            if os.path.exists(ckpt_path):
                self.model = load_model(ckpt_path, self.device)
                print(f"Loaded DUSt3R checkpoint from {ckpt_path}")
            else:
                raise FileNotFoundError(f"DUSt3R checkpoint not found: {ckpt_path}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct point cloud from images
        
        Args:
            inputs: dict with "images" (N, H, W, 3) numpy array
            
        Returns:
            dict with point cloud data and camera poses
        """
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        
        images = inputs["images"]
        
        # Save images temporarily for DUSt3R
        temp_dir = inputs.get("temp_dir", "/tmp/text2gs_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        for i, img in enumerate(images):
            path = os.path.join(temp_dir, f"view_{i:02d}.png")
            if isinstance(img, np.ndarray):
                Image.fromarray(img).save(path)
            else:
                Image.fromarray(img.cpu().numpy()).save(path)
            image_paths.append(path)
        
        # Load in DUSt3R format (API changed in newer versions)
        try:
            dust3r_images = load_images(image_paths, size=512, force_1024=False)
        except TypeError:
            # New API doesn't have force_1024
            dust3r_images = load_images(image_paths, size=512)
        
        # Run inference
        pairs = make_pairs(dust3r_images, scene_graph="complete", 
                          prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, 
                          batch_size=self.batch_size)
        
        # Global alignment
        scene = global_aligner(
            output, 
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer
        )
        loss = scene.compute_global_alignment(
            init="mst", 
            niter=self.niter,
            schedule=self.schedule, 
            lr=self.lr
        )
        
        # Extract results
        pts3d = [p.detach() for p in scene.get_pts3d()]
        c2ws = scene.get_im_poses().detach()
        focals = scene.get_focals().detach()
        principal_points = scene.get_principal_points().detach()
        depths = [d.detach() for d in scene.get_depthmaps()]
        
        # Get masks for cleaner point cloud
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.min_conf_thr)))
        masks = scene.get_masks()
        
        return {
            "pts3d": pts3d,
            "images": np.array(scene.imgs),
            "c2ws": c2ws,
            "focals": focals,
            "principal_points": principal_points,
            "depths": depths,
            "masks": masks,
            "scene": scene,
            "dust3r_images": dust3r_images,
            "image_shape": dust3r_images[0]["true_shape"],
        }
