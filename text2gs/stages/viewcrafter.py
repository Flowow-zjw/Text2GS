"""
Stage 3: ViewCrafter - Dense view synthesis with DUSt3R reconstruction
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image

from .base import BaseStage


class ViewCrafterStage(BaseStage):
    """
    Generate dense novel views using ViewCrafter and reconstruct with DUSt3R.
    
    Workflow:
    1. Use ViewCrafter to interpolate between adjacent input views
    2. Uniformly sample frames from generated videos
    3. Combine sampled frames with original frames
    4. Run DUSt3R reconstruction on combined frames
    5. Output new point cloud and camera poses for training
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        super().__init__(config, device)
        self.video_length = config.get("video_length", 25)
        self.ddim_steps = config.get("ddim_steps", 50)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.min_conf_thr = config.get("min_conf_thr", 3.0)
        self.bg_trd = config.get("bg_trd", 0.2)
        
        # 采样配置
        self.num_sampled_frames = config.get("num_sampled_frames", None)  # 目标采样帧数（不含原始帧）
        self.sample_rate = config.get("sample_rate", 6)  # 采样间隔（如果未指定num_sampled_frames）
        
        # ViewCrafter_25_sparse supports 576x1024
        self.target_height = 576
        self.target_width = 1024
        
        # DUSt3R parameters
        self.dust3r_batch_size = config.get("batch_size", 1)
        self.dust3r_niter = config.get("niter", 300)
        self.dust3r_lr = config.get("lr", 0.01)
        self.dust3r_schedule = config.get("schedule", "linear")
        
        self.diffusion_model = None
        self.dust3r_model = None
        
    def load_model(self) -> None:
        """Load ViewCrafter diffusion model (DUSt3R will be loaded later)"""
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        sys.path.insert(0, os.path.join(viewcrafter_path, "extern/dust3r"))
        
        # Load ViewCrafter
        from omegaconf import OmegaConf
        from utils.diffusion_utils import instantiate_from_config, load_model_checkpoint
        
        config_path = self.config.get(
            "config", 
            os.path.join(viewcrafter_path, "configs/inference_pvd_1024.yaml")
        )
        
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        
        self.diffusion_model = instantiate_from_config(model_config)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.cond_stage_model.device = self.device
        self.diffusion_model.perframe_ae = True
        
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
        
        # DUSt3R will be loaded later when needed
        self.dust3r_model = None
        print("  DUSt3R will be loaded after ViewCrafter completes to save memory")
    
    def _load_dust3r(self) -> None:
        """Load DUSt3R model (called after ViewCrafter is done)"""
        dust3r_path = self.config.get("dust3r_path", "./extern/dust3r")
        sys.path.insert(0, dust3r_path)
        
        print("  Loading DUSt3R model for reconstruction...")
        
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            dust3r_ckpt = self.config.get(
                "dust3r_checkpoint",
                "./checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
            if not os.path.exists(dust3r_ckpt):
                raise FileNotFoundError(f"DUSt3R checkpoint not found: {dust3r_ckpt}")
            
            self.dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_ckpt).to(self.device)
            print(f"  Loaded DUSt3R checkpoint from {dust3r_ckpt}")
        except (ImportError, AttributeError):
            from dust3r.inference import load_model
            dust3r_ckpt = self.config.get(
                "dust3r_checkpoint",
                "./checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
            if os.path.exists(dust3r_ckpt):
                self.dust3r_model = load_model(dust3r_ckpt, self.device)
                print(f"  Loaded DUSt3R checkpoint from {dust3r_ckpt}")
            else:
                raise FileNotFoundError(f"DUSt3R checkpoint not found: {dust3r_ckpt}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dense views and reconstruct with DUSt3R.
        
        Workflow:
        1. Interpolate between adjacent views using ViewCrafter
        2. Sample frames uniformly from generated videos
        3. Combine sampled + original frames
        4. Run DUSt3R reconstruction
        
        Args:
            inputs: dict with point cloud data from Stage 2
            
        Returns:
            dict with reconstructed point cloud and camera poses
        """
        viewcrafter_path = self.config.get("viewcrafter_path", "./extern/ViewCrafter")
        sys.path.insert(0, viewcrafter_path)
        from utils.pvd_utils import generate_traj_interp, setup_renderer, get_input_dict
        from pytorch3d.renderer import PerspectiveCameras
        from dust3r.utils.device import to_numpy
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.utils.image import load_images
        
        # Get inputs from Stage 2
        pts3d = inputs["pts3d"]
        c2ws = inputs["c2ws"]
        focals = inputs["focals"]
        principal_points = inputs["principal_points"]
        dust3r_images = inputs["dust3r_images"]
        original_images = inputs.get("original_images")  # High-res from MVDiffusion
        
        shape = inputs["image_shape"]
        H, W = int(shape[0][0]), int(shape[0][1])
        
        # Get masks for cleaner point cloud
        masks = inputs.get("masks")
        if masks is not None:
            masks = to_numpy(masks)
            # Background mask
            depths = inputs.get("depths", [])
            if depths:
                bgs_mask = [dpt > self.bg_trd * (torch.max(dpt[40:-40, :]) + torch.min(dpt[40:-40, :])) 
                           for dpt in depths]
                # Convert tensors to numpy and use logical OR for mask combination
                bgs_mask = to_numpy(bgs_mask)
                masks = [np.logical_or(m, mb) for m, mb in zip(masks, bgs_mask)]
        
        imgs = np.array(inputs["images"])
        num_input_views = len(imgs)
        
        # Save a copy of imgs for later use (before deletion)
        imgs_for_keyframes = imgs.copy()
        
        print(f"  Interpolating between {num_input_views} views...")
        
        # Step 1: Generate interpolated camera trajectory
        camera_traj, num_views = generate_traj_interp(
            c2ws, H, W, focals, principal_points, 
            self.video_length, self.device
        )
        
        # Step 2: Render point cloud along trajectory
        print(f"  Rendering {num_views} views from point cloud...")
        render_results = self._render_pointcloud(
            pts3d, imgs, masks, H, W, camera_traj, num_views
        )
        
        # Clean up after rendering
        del pts3d, imgs
        if masks is not None:
            del masks
        torch.cuda.empty_cache()
        
        # Resize to ViewCrafter input size (576x1024)
        render_results = F.interpolate(
            render_results.permute(0, 3, 1, 2),
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        # Update camera intrinsics for resized resolution
        # Camera pose (R, T) stays the same, but intrinsics must scale
        scale_w = self.target_width / W
        scale_h = self.target_height / H
        scaled_focals = focals * torch.tensor([[scale_w, scale_h]], device=focals.device)
        scaled_principal_points = principal_points * torch.tensor([[scale_w, scale_h]], device=principal_points.device)
        
        # Replace key frames with Stage 2 DUSt3R processed images
        # 使用Stage 2的输出而不是MVDiffusion原图
        # 优点：与点云完全一致，避免分辨率不匹配
        if imgs_for_keyframes is not None and len(imgs_for_keyframes) > 0:
            img_stage2_list = self._prepare_stage2_images(imgs_for_keyframes)
            for i in range(num_input_views):
                frame_idx = i * (self.video_length - 1)
                if frame_idx < render_results.shape[0]:
                    render_results[frame_idx] = img_stage2_list[i]
        
        # Step 3: Run ViewCrafter diffusion on each clip
        all_diffusion_results = []
        num_clips = num_input_views - 1
        
        print(f"  Generating {num_clips} video clips...")
        for clip_idx in range(num_clips):
            # Clear cache before each clip
            torch.cuda.empty_cache()
            
            start_idx = clip_idx * (self.video_length - 1)
            end_idx = start_idx + self.video_length
            
            clip_input = render_results[start_idx:end_idx]
            
            print(f"    Clip {clip_idx + 1}/{num_clips}...")
            clip_output = self._run_diffusion(clip_input)
            
            # Move to CPU immediately and clear CUDA cache
            all_diffusion_results.append(clip_output.cpu())
            del clip_output, clip_input
            torch.cuda.empty_cache()
        
        # Concatenate all clips and remove duplicates
        all_views = self._concatenate_clips_without_duplicates(
            all_diffusion_results, num_input_views
        )
        
        # ViewCrafter is done, unload it to free memory before loading DUSt3R
        print("  ViewCrafter generation complete, unloading model...")
        del self.diffusion_model
        self.diffusion_model = None
        torch.cuda.empty_cache()
        print(f"  Freed ViewCrafter memory")
        
        # Step 4: Sample frames uniformly (保留原始帧)
        original_frame_indices = [i * (self.video_length - 1) for i in range(num_input_views)]
        
        if self.num_sampled_frames is not None:
            print(f"  Sampling {self.num_sampled_frames} frames (保留{num_input_views}个原始帧)...")
            sampled_frames, sampled_indices = self._sample_frames_by_count(
                all_views, self.num_sampled_frames, original_frame_indices
            )
        else:
            print(f"  Sampling frames (rate={self.sample_rate}, 保留原始帧)...")
            sampled_frames, sampled_indices = self._sample_frames_by_rate(
                all_views, self.sample_rate, original_frame_indices
            )
        
        # Step 5: Prepare images for DUSt3R reconstruction
        print(f"  Preparing {len(sampled_frames)} frames for DUSt3R reconstruction...")
        combined_images = self._prepare_dust3r_images(
            sampled_frames, None, inputs.get("temp_dir", "/tmp/text2gs_temp")  # 不传 original_images，避免重复
        )
        
        # Load DUSt3R model now (after ViewCrafter is unloaded)
        if self.dust3r_model is None:
            self._load_dust3r()
        
        # Step 6: Run DUSt3R reconstruction
        print(f"  Running DUSt3R reconstruction on {len(combined_images)} images...")
        reconstructed_data = self._run_dust3r_reconstruction(combined_images)
        
        # Add metadata
        reconstructed_data["all_views"] = [all_views]  # Keep for visualization
        reconstructed_data["original_images"] = original_images
        reconstructed_data["num_input_views"] = num_input_views
        reconstructed_data["video_length"] = self.video_length
        reconstructed_data["num_sampled_frames"] = len(sampled_frames)
        reconstructed_data["sampled_indices"] = sampled_indices
        reconstructed_data["original_frame_indices"] = original_frame_indices
        
        # Add scaled camera parameters for ViewCrafter generated frames
        reconstructed_data["viewcrafter_focals"] = scaled_focals
        reconstructed_data["viewcrafter_principal_points"] = scaled_principal_points
        reconstructed_data["viewcrafter_resolution"] = (self.target_height, self.target_width)
        
        return reconstructed_data
    
    def _concatenate_clips_without_duplicates(
        self, clips: List[torch.Tensor], num_input_views: int
    ) -> torch.Tensor:
        """
        拼接clips并去除重复的边界帧
        
        Args:
            clips: List of clip tensors, each (25, H, W, 3)
            num_input_views: Number of original input views
            
        Returns:
            Concatenated tensor without duplicate boundary frames
        """
        if len(clips) == 0:
            return torch.empty(0)
        
        # 第一个clip保留全部
        result = [clips[0]]
        
        # 后续clips跳过第一帧（因为与前一个clip的最后一帧重复）
        for clip in clips[1:]:
            result.append(clip[1:])
        
        concatenated = torch.cat(result, dim=0)
        
        # 计算去重后的总帧数
        # 第一个clip: 25帧
        # 后续clips: 每个24帧（跳过第一帧）
        # 总计: 25 + (num_clips - 1) * 24 = 25 + (num_input_views - 2) * 24
        expected_frames = 25 + (num_input_views - 2) * 24
        
        print(f"  Removed {len(clips) * 25 - concatenated.shape[0]} duplicate frames")
        print(f"  Total frames after deduplication: {concatenated.shape[0]} (expected: {expected_frames})")
        
        return concatenated
    
    def _sample_frames_by_count(
        self, 
        all_views: torch.Tensor, 
        num_samples: int,
        original_indices: List[int]
    ) -> tuple:
        """
        按指定数量采样帧（保留所有原始帧）
        
        Args:
            all_views: (N, H, W, 3) all generated frames
            num_samples: 目标采样帧数（不包括原始帧）
            original_indices: 原始帧的索引列表
            
        Returns:
            (sampled_frames, sampled_indices)
        """
        total_frames = all_views.shape[0]
        original_set = set(original_indices)
        
        # 可采样的帧索引（排除原始帧）
        available_indices = [i for i in range(total_frames) if i not in original_set]
        
        if num_samples >= len(available_indices):
            # 如果要求的采样数大于等于可用帧数，全部采样
            sampled_non_original = available_indices
        else:
            # 均匀采样
            step = len(available_indices) / num_samples
            sampled_non_original = [
                available_indices[int(i * step)] 
                for i in range(num_samples)
            ]
        
        # 合并原始帧和采样帧，并排序
        all_sampled_indices = sorted(list(original_set) + sampled_non_original)
        
        # 提取帧
        sampled_frames = [all_views[i] for i in all_sampled_indices]
        
        print(f"    原始帧: {len(original_indices)}个")
        print(f"    采样帧: {len(sampled_non_original)}个")
        print(f"    总计: {len(all_sampled_indices)}个")
        
        return sampled_frames, all_sampled_indices
    
    def _sample_frames_by_rate(
        self, 
        all_views: torch.Tensor, 
        sample_rate: int,
        original_indices: List[int]
    ) -> tuple:
        """
        按采样率采样帧（保留所有原始帧）
        
        Args:
            all_views: (N, H, W, 3) all generated frames
            sample_rate: 采样间隔
            original_indices: 原始帧的索引列表
            
        Returns:
            (sampled_frames, sampled_indices)
        """
        total_frames = all_views.shape[0]
        original_set = set(original_indices)
        
        # 按间隔采样（排除原始帧位置）
        sampled_non_original = []
        for i in range(0, total_frames, sample_rate):
            if i not in original_set:
                sampled_non_original.append(i)
        
        # 合并原始帧和采样帧，并排序
        all_sampled_indices = sorted(list(original_set) + sampled_non_original)
        
        # 提取帧
        sampled_frames = [all_views[i] for i in all_sampled_indices]
        
        print(f"    原始帧: {len(original_indices)}个")
        print(f"    采样帧: {len(sampled_non_original)}个")
        print(f"    总计: {len(all_sampled_indices)}个")
        
        return sampled_frames, all_sampled_indices
        reconstructed_data["num_sampled_frames"] = len(sampled_frames)
        
        return reconstructed_data

    def _sample_frames(self, all_views: torch.Tensor, sample_rate: int) -> List[torch.Tensor]:
        """Uniformly sample frames from generated videos"""
        sampled = []
        for i in range(0, all_views.shape[0], sample_rate):
            sampled.append(all_views[i])
        return sampled
    
    def _prepare_dust3r_images(self, sampled_frames: List[torch.Tensor], 
                               original_images: np.ndarray, temp_dir: str) -> List:
        """
        Prepare images for DUSt3R reconstruction.
        
        Args:
            sampled_frames: Sampled frames from ViewCrafter (576x1024, [-1,1])
            original_images: Original images from MVDiffusion (512x512, [0,255])
            temp_dir: Temporary directory for saving images
            
        Returns:
            List of DUSt3R image dicts
        """
        from dust3r.utils.image import load_images
        
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save original images (resize to DUSt3R size 512x384)
        image_paths = []
        
        # Add original images first
        if original_images is not None:
            for i, img in enumerate(original_images):
                path = os.path.join(temp_dir, f"orig_{i:04d}.png")
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img if img.max() > 1 else (img * 255).astype(np.uint8))
                else:
                    pil_img = Image.fromarray(img.cpu().numpy() if img.max() > 1 else (img.cpu().numpy() * 255).astype(np.uint8))
                pil_img.save(path)
                image_paths.append(path)
        
        # Add sampled frames
        for i, frame in enumerate(sampled_frames):
            path = os.path.join(temp_dir, f"sampled_{i:04d}.png")
            # Convert from [-1,1] to [0,255]
            if hasattr(frame, 'cpu'):
                frame_np = ((frame.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            else:
                frame_np = ((frame + 1) / 2 * 255).astype(np.uint8)
            Image.fromarray(frame_np).save(path)
            image_paths.append(path)
        
        # Load with DUSt3R
        try:
            dust3r_images = load_images(image_paths, size=512, force_1024=False)
        except TypeError:
            dust3r_images = load_images(image_paths, size=512)
        
        return dust3r_images
    
    def _run_dust3r_reconstruction(self, images: List) -> Dict[str, Any]:
        """Run DUSt3R reconstruction on combined images"""
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        
        # Run inference
        pairs = make_pairs(images, scene_graph="complete", 
                          prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r_model, self.device, 
                          batch_size=self.dust3r_batch_size)
        
        # Global alignment
        scene = global_aligner(
            output, 
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer
        )
        loss = scene.compute_global_alignment(
            init="mst", 
            niter=self.dust3r_niter,
            schedule=self.dust3r_schedule, 
            lr=self.dust3r_lr
        )
        
        # Extract results
        pts3d = [p.detach() for p in scene.get_pts3d()]
        c2ws = scene.get_im_poses().detach()
        focals = scene.get_focals().detach()
        principal_points = scene.get_principal_points().detach()
        depths = [d.detach() for d in scene.get_depthmaps()]
        
        # Get masks
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.min_conf_thr)))
        masks = scene.get_masks()
        
        # Extract all needed data before deleting scene
        result = {
            "pts3d": pts3d,
            "images": np.array(scene.imgs),
            "c2ws": c2ws,
            "focals": focals,
            "principal_points": principal_points,
            "depths": depths,
            "masks": masks,
            "dust3r_images": images,
            "image_shape": images[0]["true_shape"],
        }
        
        # Delete scene to free memory immediately
        del scene
        torch.cuda.empty_cache()
        
        return result
    
    def _prepare_stage2_images(self, stage2_images: np.ndarray) -> List[torch.Tensor]:
        """
        Prepare Stage 2 DUSt3R processed images for key frames
        
        Args:
            stage2_images: Stage 2输出的图像 (N, 384, 512, 3) [0, 1]
            
        Returns:
            List of tensors resized to ViewCrafter resolution
        """
        img_list = []
        for img in stage2_images:
            # 确保是[0, 1]范围
            if img.max() > 1:
                img = img / 255.0
            
            # 转换为tensor
            img_tensor = torch.from_numpy(img).float().to(self.device)
            
            # Resize到ViewCrafter分辨率 (384, 512) → (576, 1024)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 384, 512)
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.target_height, self.target_width),  # (576, 1024)
                mode="bilinear",
                align_corners=False
            )
            img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)  # (576, 1024, 3)
            
            img_list.append(img_tensor)
        
        return img_list
    
    def _prepare_original_images(self, original_images: np.ndarray) -> List[torch.Tensor]:
        """Prepare original high-res images for key frames"""
        img_ori_list = []
        for img in original_images:
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
    
    def _render_pointcloud(self, pts3d, imgs, masks, H, W, cameras, num_views):
        """Render point cloud from camera viewpoints"""
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import (
            PointsRasterizationSettings, PointsRenderer,
            PointsRasterizer, AlphaCompositor, PerspectiveCameras
        )
        from dust3r.utils.device import to_numpy
        
        pts3d_np = to_numpy(pts3d)
        
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
        
        # Render in batches
        batch_size = 25
        all_renders = []
        
        for start_idx in range(0, num_views, batch_size):
            end_idx = min(start_idx + batch_size, num_views)
            batch_views = end_idx - start_idx
            
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
        
        # Clean up intermediate tensors
        del videos, batch_samples
        torch.cuda.empty_cache()
        
        return result
    
    def unload_model(self) -> None:
        """Unload both ViewCrafter and DUSt3R models to free memory"""
        if self.diffusion_model is not None:
            del self.diffusion_model
            self.diffusion_model = None
            print("  Unloaded ViewCrafter diffusion model")
        
        if self.dust3r_model is not None:
            del self.dust3r_model
            self.dust3r_model = None
            print("  Unloaded DUSt3R model")
        
        torch.cuda.empty_cache()
        print("  Freed Stage 3 memory")
