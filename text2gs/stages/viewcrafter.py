"""
Stage 3: ViewCrafter - Dense view synthesis using sparse_view_interp mode
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional

from .base import BaseStage


class ViewCrafterStage(BaseStage):
    """
    Generate dense novel views using ViewCrafter sparse_view_interp mode.
    
    This mode interpolates between all adjacent input views to generate
    a dense 360° coverage with minimal angular gaps.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        super().__init__(config, device)
        self.video_length = config.get("video_length", 25)
        self.ddim_steps = config.get("ddim_steps", 50)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.min_conf_thr = config.get("min_conf_thr", 3.0)
        self.bg_trd = config.get("bg_trd", 0.2)
        
        # ViewCrafter_25_sparse only supports 576x1024
        # 512 model doesn't have sparse version
        self.target_height = 576
        self.target_width = 1024
        
        self.diffusion_model = None
        
    def load_model(self) -> None:
        """Load ViewCrafter diffusion model"""
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        sys.path.insert(0, os.path.join(viewcrafter_path, "extern/dust3r"))
        
        from omegaconf import OmegaConf
        from utils.diffusion_utils import instantiate_from_config, load_model_checkpoint
        
        # Use 1024 config for sparse view interpolation
        config_path = self.config.get(
            "config", 
            os.path.join(viewcrafter_path, "configs/inference_pvd_1024.yaml")
        )
        
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        
        # Create model
        self.diffusion_model = instantiate_from_config(model_config)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.cond_stage_model.device = self.device
        self.diffusion_model.perframe_ae = True
        
        # Load checkpoint
        ckpt_path = self.config.get(
            "checkpoint",
            "./checkpoints/viewcrafter/model_sparse.ckpt"
        )
        if os.path.exists(ckpt_path):
            self.diffusion_model = load_model_checkpoint(self.diffusion_model, ckpt_path)
            print(f"Loaded ViewCrafter checkpoint from {ckpt_path}")
        else:
            raise FileNotFoundError(f"ViewCrafter checkpoint not found: {ckpt_path}")
        
        self.diffusion_model.eval()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dense views using sparse_view_interp mode.
        
        Args:
            inputs: dict with point cloud data from Stage 2
            
        Returns:
            dict with all generated views and camera poses
        """
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        from utils.pvd_utils import (
            setup_renderer, save_video, 
            interpolate_poses_spline, interpolate_sequence
        )
        from pytorch3d.renderer import PerspectiveCameras
        from dust3r.utils.device import to_numpy
        
        # Get inputs
        pts3d = inputs["pts3d"]
        c2ws = inputs["c2ws"]
        focals = inputs["focals"]
        principal_points = inputs["principal_points"]
        dust3r_images = inputs["dust3r_images"]
        original_images = inputs.get("original_images")  # High-res originals from MVDiffusion
        
        shape = inputs["image_shape"]
        H, W = int(shape[0][0]), int(shape[0][1])
        
        # Get masks for cleaner point cloud
        masks = inputs.get("masks")
        if masks is not None:
            masks = to_numpy(masks)
        
        # Get images
        imgs = np.array([img for img in inputs["images"]])
        
        # Prepare high-resolution original images for key frames
        if original_images is not None:
            img_ori_list = self._prepare_original_images(original_images)
        else:
            # Fallback: resize dust3r images
            img_ori_list = self._prepare_fallback_images(imgs)
        
        num_input_views = len(imgs)
        print(f"  Interpolating between {num_input_views} views...")
        
        # Generate interpolated camera trajectory
        camera_traj, num_views, c2ws_interp = self._generate_interp_trajectory(
            c2ws, H, W, focals, principal_points
        )
        
        # Render point cloud along trajectory
        print(f"  Rendering {num_views} views from point cloud...")
        render_results = self._render_pointcloud(
            pts3d, imgs, masks, H, W, camera_traj, num_views
        )
        
        # Resize to ViewCrafter input size
        render_results = F.interpolate(
            render_results.permute(0, 3, 1, 2),
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        # Replace key frames with original high-res images
        for i in range(num_input_views):
            frame_idx = i * (self.video_length - 1)
            if frame_idx < render_results.shape[0]:
                render_results[frame_idx] = img_ori_list[i]
        
        # Generate clips between adjacent views
        all_diffusion_results = []
        num_clips = num_input_views - 1
        
        print(f"  Generating {num_clips} video clips...")
        for clip_idx in range(num_clips):
            start_idx = clip_idx * (self.video_length - 1)
            end_idx = start_idx + self.video_length
            
            clip_input = render_results[start_idx:end_idx]
            
            print(f"    Clip {clip_idx + 1}/{num_clips}...")
            clip_output = self._run_diffusion(clip_input)
            
            # Move to CPU to save memory
            all_diffusion_results.append(clip_output.cpu())
            
            torch.cuda.empty_cache()
        
        # Concatenate all clips
        all_views = torch.cat(all_diffusion_results, dim=0)
        
        return {
            "all_views": [all_views],  # Keep as list for compatibility
            "pts3d": pts3d,
            "images": imgs,
            "original_images": original_images,  # Pass through for Stage 4
            "c2ws": c2ws,
            "c2ws_interp": c2ws_interp,
            "focals": focals,
            "principal_points": principal_points,
            "masks": masks,
            "num_input_views": num_input_views,
            "video_length": self.video_length,
        }

    def _prepare_original_images(self, original_images: np.ndarray) -> List[torch.Tensor]:
        """Prepare original high-res images for key frames"""
        img_ori_list = []
        for img in original_images:
            # Convert to tensor [H, W, 3] in [0, 1]
            if img.max() > 1:
                img = img / 255.0
            img_tensor = torch.from_numpy(img).float().to(self.device)
            
            # Resize to target size (320x512 for 512 model)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False
            )
            img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)
            img_ori_list.append(img_tensor)
        
        return img_ori_list
    
    def _prepare_fallback_images(self, imgs: np.ndarray) -> List[torch.Tensor]:
        """Fallback: prepare images from dust3r output"""
        img_ori_list = []
        for img in imgs:
            if img.max() > 1:
                img = img / 255.0
            img_tensor = torch.from_numpy(img).float().to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False
            )
            img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)
            img_ori_list.append(img_tensor)
        return img_ori_list
    
    def _generate_interp_trajectory(self, c2ws, H, W, focals, principal_points):
        """Generate interpolated camera trajectory between all views (same as ViewCrafter)"""
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        from utils.pvd_utils import interpolate_sequence
        from pytorch3d.renderer import PerspectiveCameras
        
        # Ensure tensors are on device
        c2ws_device = c2ws.to(self.device) if c2ws.device.type == 'cpu' else c2ws
        focals_device = focals.to(self.device) if focals.device.type == 'cpu' else focals
        pp_device = principal_points.to(self.device) if principal_points.device.type == 'cpu' else principal_points
        
        # Interpolate poses between adjacent views
        c2ws_interp = self._interp_poses(c2ws_device, n_inserts=self.video_length)
        num_views = c2ws_interp.shape[0]
        
        # Convert to PyTorch3D format
        R, T = c2ws_interp[:, :3, :3], c2ws_interp[:, :3, 3:]
        R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
        new_c2w = torch.cat([R, T], 2)
        
        w2c = torch.linalg.inv(torch.cat(
            (new_c2w, torch.tensor([[[0, 0, 0, 1]]]).to(self.device).repeat(num_views, 1, 1)), 1
        ))
        R_new = w2c[:, :3, :3].permute(0, 2, 1)
        T_new = w2c[:, :3, 3]
        
        # Interpolate camera intrinsics using ViewCrafter's function
        focals_interp = interpolate_sequence(focals_device, self.video_length - 2, self.device)
        pp_interp = interpolate_sequence(pp_device, self.video_length - 2, self.device)
        
        cameras = PerspectiveCameras(
            focal_length=focals_interp,
            principal_point=pp_interp,
            in_ndc=False,
            image_size=((H, W),),
            R=R_new,
            T=T_new,
            device=self.device
        )
        
        return cameras, num_views, c2ws_interp
    
    def _interp_poses(self, c2ws: torch.Tensor, n_inserts: int = 25) -> torch.Tensor:
        """Interpolate poses between adjacent views using spline (same as ViewCrafter)"""
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        from utils.pvd_utils import interpolate_poses_spline
        
        n_poses = c2ws.shape[0]
        interpolated_poses = []
        
        for i in range(n_poses - 1):
            start_pose = c2ws[i]
            end_pose = c2ws[i + 1]
            
            # Use ViewCrafter's spline interpolation
            interpolated_path = interpolate_poses_spline(
                torch.stack([start_pose, end_pose])[:, :3, :].cpu().numpy(), 
                n_inserts
            ).to(self.device)
            
            # Exclude last to avoid duplicates
            interpolated_path = interpolated_path[:-1]
            interpolated_poses.append(interpolated_path)
        
        # Add final pose
        interpolated_poses.append(c2ws[-1:])
        
        return torch.cat(interpolated_poses, dim=0)
    
    def _render_pointcloud(self, pts3d, imgs, masks, H, W, cameras, num_views):
        """Render point cloud from camera viewpoints in batches to save memory"""
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import (
            PointsRasterizationSettings, PointsRenderer,
            PointsRasterizer, AlphaCompositor, PerspectiveCameras
        )
        from dust3r.utils.device import to_numpy
        
        pts3d_np = to_numpy(pts3d)
        
        # Handle imgs which may be list of different sized arrays
        if isinstance(imgs, list):
            imgs_np = [to_numpy(img) if not isinstance(img, np.ndarray) else img for img in imgs]
        else:
            imgs_np = imgs if isinstance(imgs, np.ndarray) else to_numpy(imgs)
        
        if masks is not None:
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d_np, masks)])).to(self.device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs_np, masks)])).to(self.device)
        else:
            pts = torch.from_numpy(np.concatenate([p.reshape(-1, 3) for p in pts3d_np])).to(self.device)
            col = torch.from_numpy(np.concatenate([p.reshape(-1, 3) for p in imgs_np])).to(self.device)
        
        raster_settings = PointsRasterizationSettings(
            image_size=(H, W), radius=0.01, points_per_pixel=10, bin_size=0
        )
        
        # Render in batches to save memory
        batch_size = 25  # Render 25 views at a time
        all_renders = []
        
        for start_idx in range(0, num_views, batch_size):
            end_idx = min(start_idx + batch_size, num_views)
            batch_views = end_idx - start_idx
            
            # Create batch cameras with correct image_size for batch
            batch_cameras = PerspectiveCameras(
                focal_length=cameras.focal_length[start_idx:end_idx],
                principal_point=cameras.principal_point[start_idx:end_idx],
                R=cameras.R[start_idx:end_idx],
                T=cameras.T[start_idx:end_idx],
                in_ndc=False,
                image_size=((H, W),) * batch_views,
                device=self.device
            )
            
            point_cloud = Pointclouds(points=[pts], features=[col]).extend(batch_views)
            
            renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=batch_cameras, raster_settings=raster_settings),
                compositor=AlphaCompositor()
            )
            
            batch_render = renderer(point_cloud)
            all_renders.append(batch_render.cpu())
            
            del point_cloud, renderer, batch_cameras, batch_render
            torch.cuda.empty_cache()
        
        return torch.cat(all_renders, dim=0).to(self.device)
    
    def _run_diffusion(self, renderings: torch.Tensor) -> torch.Tensor:
        """Run ViewCrafter diffusion on a single clip"""
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        from utils.diffusion_utils import image_guided_synthesis
        
        prompt = "High quality 3D scene"
        
        # Convert to diffusion input format: [0,1] -> [-1,1]
        videos = (renderings * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        
        # Noise shape
        h, w = self.target_height // 8, self.target_width // 8
        channels = self.diffusion_model.model.diffusion_model.out_channels
        noise_shape = [1, channels, renderings.shape[0], h, w]
        
        # Condition on first frame
        condition_index = [0]
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            batch_samples = image_guided_synthesis(
                self.diffusion_model,
                [prompt],
                videos,
                noise_shape,
                n_samples=1,
                ddim_steps=self.ddim_steps,
                ddim_eta=1.0,
                unconditional_guidance_scale=self.guidance_scale,
                cfg_img=None,
                fs=10,
                text_input=True,
                multiple_cond_cfg=False,
                timestep_spacing="uniform_trailing",
                guidance_rescale=0.7,
                condition_index=condition_index
            )
        
        # Output: [1, 1, C, T, H, W] -> [T, H, W, C] in [-1, 1]
        result = torch.clamp(batch_samples[0][0].permute(1, 2, 3, 0), -1.0, 1.0)
        
        return result
