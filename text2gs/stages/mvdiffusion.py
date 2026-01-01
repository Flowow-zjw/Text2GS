"""
Stage 1: MVDiffusion - Generate multi-view panoramic images from text
"""

import os
import sys
import torch
import numpy as np
import cv2
from typing import Any, Dict, List, Optional
from PIL import Image

from .base import BaseStage


class MVDiffusionStage(BaseStage):
    """Generate 8 panoramic views using MVDiffusion"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        super().__init__(config, device)
        self.num_views = config.get("num_views", 8)
        self.resolution = config.get("resolution", 512)
        self.fov = config.get("fov", 90)
        self.guidance_scale = config.get("guidance_scale", 9.0)
        self.diff_timesteps = config.get("diff_timesteps", 50)
        
    def load_model(self) -> None:
        """Load MVDiffusion model"""
        # Add MVDiffusion to path
        mvdiff_path = self.config.get("mvdiffusion_path", "./extern/MVDiffusion")
        sys.path.insert(0, mvdiff_path)
        
        import yaml
        from src.lightning_pano_gen import PanoGenerator
        
        # Load config
        config_file = os.path.join(mvdiff_path, "configs/pano_generation.yaml")
        mv_config = yaml.load(open(config_file, "rb"), Loader=yaml.SafeLoader)
        
        # Override with our settings
        mv_config["model"]["guidance_scale"] = self.guidance_scale
        mv_config["model"]["diff_timestep"] = self.diff_timesteps
        
        # Create model
        self.model = PanoGenerator(mv_config)
        
        # Load checkpoint
        ckpt_path = self.config.get("checkpoint", "./checkpoints/mvdiffusion/pano.ckpt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded MVDiffusion checkpoint from {ckpt_path}")
        else:
            raise FileNotFoundError(f"MVDiffusion checkpoint not found: {ckpt_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate panoramic views from text prompt
        
        Args:
            inputs: dict with "text" key containing the prompt
            
        Returns:
            dict with "images" (N, H, W, 3) and "cameras" (K, R matrices)
        """
        text_prompt = inputs["text"]
        
        # Generate camera parameters
        cameras = self._generate_cameras()
        
        # Prepare batch
        images = torch.zeros(
            (1, self.num_views, self.resolution, self.resolution, 3)
        ).to(self.device)
        
        prompts = [text_prompt] * self.num_views
        
        K = torch.tensor(cameras["K"]).to(self.device)[None]
        R = torch.tensor(cameras["R"]).to(self.device)[None]
        
        batch = {
            "images": images,
            "prompt": prompts,
            "R": R,
            "K": K,
        }
        
        # Generate
        with torch.no_grad():
            images_pred = self.model.inference(batch)  # (1, N, H, W, 3) uint8
        
        return {
            "images": images_pred[0],  # (N, H, W, 3)
            "cameras": cameras,
            "prompt": text_prompt,
        }
    
    def _generate_cameras(self) -> Dict[str, np.ndarray]:
        """Generate camera intrinsics and rotations for panoramic views"""
        Rs, Ks = [], []
        
        for i in range(self.num_views):
            degree = (360 // self.num_views * i) % 360
            K, R = self._get_K_R(self.fov, degree, 0)
            Rs.append(R)
            Ks.append(K)
        
        return {
            "K": np.array(Ks, dtype=np.float32),
            "R": np.array(Rs, dtype=np.float32),
            "resolution": self.resolution,
            "fov": self.fov,
        }
    
    def _get_K_R(self, fov: float, theta: float, phi: float) -> tuple:
        """Compute intrinsic K and rotation R"""
        f = 0.5 * self.resolution / np.tan(0.5 * fov / 180.0 * np.pi)
        cx = cy = (self.resolution - 1) / 2.0
        
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
        R = (R2 @ R1).astype(np.float32)
        
        return K, R
