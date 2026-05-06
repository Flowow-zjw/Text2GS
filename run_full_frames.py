#!/usr/bin/env python3
"""
独立运行脚本：使用所有帧进行DUSt3R重建

这个脚本不修改原始text2gs代码，通过monkey patching的方式扩展功能。

使用方法：
    python run_full_frames.py --text "A beautiful garden" --batch-size 30
"""

import argparse
import yaml
import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List

# 导入原始text2gs
from text2gs.pipeline import Text2GSPipeline
from text2gs.stages.viewcrafter import ViewCrafterStage


def add_batched_reconstruction_methods():
    """
    通过monkey patching为ViewCrafterStage添加批量重建方法
    """
    
    def _run_dust3r_reconstruction_batched(self, images: List, batch_size: int = 30) -> Dict[str, Any]:
        """
        批量增量式DUSt3R重建
        
        Args:
            images: 所有DUSt3R图像
            batch_size: 每批处理的图像数量
            
        Returns:
            重建结果字典
        """
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        
        num_images = len(images)
        print(f"  批量重建: {num_images}张图像, batch_size={batch_size}")
        
        if num_images <= batch_size:
            print(f"  图像数量<=batch_size, 使用标准重建")
            return self._run_dust3r_reconstruction_original(images)
        
        overlap = batch_size // 4  # 25%重叠
        
        all_pts3d = []
        all_c2ws = []
        all_focals = []
        all_principal_points = []
        all_depths = []
        all_masks = []
        all_imgs = []
        
        reference_c2ws = None
        reference_indices = None
        
        batch_idx = 0
        start_idx = 0
        
        while start_idx < num_images:
            end_idx = min(start_idx + batch_size, num_images)
            batch_images = images[start_idx:end_idx]
            
            print(f"  批次{batch_idx + 1}: 图像{start_idx}-{end_idx-1} ({len(batch_images)}张)")
            
            # 运行推理
            pairs = make_pairs(batch_images, scene_graph="complete", 
                              prefilter=None, symmetrize=True)
            output = inference(pairs, self.dust3r_model, self.device, 
                              batch_size=self.dust3r_batch_size)
            
            # 全局对齐
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
            
            # 提取结果
            batch_pts3d = [p.detach().cpu() for p in scene.get_pts3d()]
            batch_c2ws = scene.get_im_poses().detach().cpu()
            batch_focals = scene.get_focals().detach().cpu()
            batch_principal_points = scene.get_principal_points().detach().cpu()
            batch_depths = [d.detach().cpu() for d in scene.get_depthmaps()]
            
            scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.min_conf_thr)))
            batch_masks = scene.get_masks()
            batch_imgs = np.array(scene.imgs)
            
            del scene, output, pairs
            torch.cuda.empty_cache()
            
            if batch_idx == 0:
                # 第一批：建立参考坐标系
                reference_c2ws = batch_c2ws.clone()
                reference_indices = list(range(start_idx, end_idx))
                
                all_pts3d.extend(batch_pts3d)
                all_c2ws.append(batch_c2ws)
                all_focals.append(batch_focals)
                all_principal_points.append(batch_principal_points)
                all_depths.extend(batch_depths)
                all_masks.extend(batch_masks)
                all_imgs.append(batch_imgs)
                
                print(f"    建立参考坐标系: {len(batch_c2ws)}个相机")
            else:
                # 后续批次：对齐到参考坐标系
                overlap_start = max(0, start_idx - overlap)
                overlap_end = start_idx
                
                if overlap_start < overlap_end:
                    ref_overlap_indices = [i - reference_indices[0] for i in range(overlap_start, overlap_end) 
                                          if i in reference_indices]
                    curr_overlap_indices = [i - start_idx for i in range(overlap_start, overlap_end)]
                    
                    if len(ref_overlap_indices) > 0 and len(curr_overlap_indices) > 0:
                        ref_poses = reference_c2ws[ref_overlap_indices]
                        curr_poses = batch_c2ws[curr_overlap_indices]
                        
                        # Procrustes对齐
                        transform_matrix = compute_alignment_transform(
                            curr_poses[:, :3, 3].numpy(),
                            ref_poses[:, :3, 3].numpy()
                        )
                        
                        print(f"    对齐到参考坐标系: {len(ref_overlap_indices)}个重叠相机")
                        
                        # 应用变换
                        batch_c2ws = apply_transform_to_poses(batch_c2ws, transform_matrix)
                        batch_pts3d = [apply_transform_to_points(pts, transform_matrix) 
                                      for pts in batch_pts3d]
                
                # 存储非重叠部分
                non_overlap_start = overlap if start_idx > 0 else 0
                all_pts3d.extend(batch_pts3d[non_overlap_start:])
                all_c2ws.append(batch_c2ws[non_overlap_start:])
                all_focals.append(batch_focals[non_overlap_start:])
                all_principal_points.append(batch_principal_points[non_overlap_start:])
                all_depths.extend(batch_depths[non_overlap_start:])
                all_masks.extend(batch_masks[non_overlap_start:])
                all_imgs.append(batch_imgs[non_overlap_start:])
                
                reference_c2ws = batch_c2ws.clone()
                reference_indices = list(range(start_idx, end_idx))
            
            batch_idx += 1
            start_idx = end_idx - overlap if end_idx < num_images else num_images
        
        # 合并结果
        final_c2ws = torch.cat(all_c2ws, dim=0).to(self.device)
        final_focals = torch.cat(all_focals, dim=0).to(self.device)
        final_principal_points = torch.cat(all_principal_points, dim=0).to(self.device)
        final_imgs = np.concatenate(all_imgs, axis=0)
        
        all_pts3d = [pts.to(self.device) for pts in all_pts3d]
        all_depths = [d.to(self.device) for d in all_depths]
        
        print(f"  批量重建完成: {len(all_pts3d)}张图像, {len(final_c2ws)}个相机")
        
        return {
            "pts3d": all_pts3d,
            "images": final_imgs,
            "c2ws": final_c2ws,
            "focals": final_focals,
            "principal_points": final_principal_points,
            "depths": all_depths,
            "masks": all_masks,
            "dust3r_images": images,
            "image_shape": images[0]["true_shape"],
        }
    
    # 保存原始方法
    ViewCrafterStage._run_dust3r_reconstruction_original = ViewCrafterStage._run_dust3r_reconstruction
    
    # 添加新方法
    ViewCrafterStage._run_dust3r_reconstruction_batched = _run_dust3r_reconstruction_batched


def compute_alignment_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Procrustes对齐"""
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = target_center - R @ source_center
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    return transform


def apply_transform_to_poses(poses: torch.Tensor, transform: np.ndarray) -> torch.Tensor:
    """应用变换到位姿"""
    transform_tensor = torch.from_numpy(transform).float().to(poses.device)
    return transform_tensor @ poses


def apply_transform_to_points(points: torch.Tensor, transform: np.ndarray) -> torch.Tensor:
    """应用变换到点云"""
    original_shape = points.shape
    pts = points.reshape(-1, 3)
    
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    pts_homo = torch.cat([pts, ones], dim=1)
    
    transform_tensor = torch.from_numpy(transform).float().to(pts.device)
    pts_transformed = (transform_tensor @ pts_homo.T).T
    
    return pts_transformed[:, :3].reshape(original_shape)


def patch_viewcrafter_run_method(use_all_frames: bool, batch_size: int):
    """
    Patch ViewCrafterStage的run方法以支持所有帧模式
    """
    original_run = ViewCrafterStage.run
    
    def patched_run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 调用原始run方法获取所有生成的帧
        result = original_run(self, inputs)
        
        if not use_all_frames:
            return result
        
        # 使用所有帧模式
        print(f"\n  [扩展功能] 使用所有帧进行重建...")
        
        # 获取所有生成的帧
        all_views = result.get("all_views", [None])[0]
        if all_views is None:
            print("  警告: 未找到all_views，使用标准模式")
            return result
        
        # 准备所有帧
        all_frame_list = [all_views[i] for i in range(all_views.shape[0])]
        
        # 准备DUSt3R图像
        from dust3r.utils.image import load_images
        from PIL import Image
        
        temp_dir = inputs.get("temp_dir", "/tmp/text2gs_temp_full")
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        for i, frame in enumerate(all_frame_list):
            path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            if hasattr(frame, 'cpu'):
                frame_np = ((frame.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            else:
                frame_np = ((frame + 1) / 2 * 255).astype(np.uint8)
            Image.fromarray(frame_np).save(path)
            image_paths.append(path)
        
        try:
            dust3r_images = load_images(image_paths, size=512, force_1024=False)
        except TypeError:
            dust3r_images = load_images(image_paths, size=512)
        
        # 加载DUSt3R模型
        if self.dust3r_model is None:
            self._load_dust3r()
        
        # 批量重建
        reconstructed_data = self._run_dust3r_reconstruction_batched(dust3r_images, batch_size)
        
        # 更新结果
        reconstructed_data["all_views"] = [all_views]
        reconstructed_data["num_sampled_frames"] = len(all_frame_list)
        reconstructed_data["sampled_indices"] = list(range(len(all_frame_list)))
        
        return reconstructed_data
    
    ViewCrafterStage.run = patched_run


def main():
    parser = argparse.ArgumentParser(description="使用所有帧进行DUSt3R重建")
    parser.add_argument("--text", type=str, required=True, help="文本提示词")
    parser.add_argument("--config", type=str, default="configs/full_frames.yaml", help="配置文件")
    parser.add_argument("--output", type=str, default="./output_full_frames", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=30, help="批量重建的batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--compress", action="store_true", help="压缩输出的3DGS模型")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    config["output_dir"] = args.output
    config["device"] = args.device
    
    # 设置压缩选项
    if args.compress:
        if "gaussian" not in config:
            config["gaussian"] = {}
        config["gaussian"]["compress"] = True
    
    print("=" * 80)
    print("Text2GS - 所有帧重建模式（扩展版）")
    print("=" * 80)
    print(f"提示词: {args.text}")
    print(f"批量大小: {args.batch_size}")
    print(f"输出目录: {args.output}")
    print(f"压缩模式: {'✓ 启用' if args.compress else '✗ 禁用'}")
    print("=" * 80)
    
    # 应用扩展功能
    print("\n[1] 添加批量重建方法...")
    add_batched_reconstruction_methods()
    
    print("[2] Patch ViewCrafter run方法...")
    patch_viewcrafter_run_method(use_all_frames=True, batch_size=args.batch_size)
    
    print("[3] 创建pipeline...")
    pipeline = Text2GSPipeline(config)
    
    print("[4] 运行pipeline...\n")
    results = pipeline.run(text=args.text, save_intermediate=True)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"输出目录: {results['output_dir']}")
    
    # 统计信息
    stage3 = results.get("stage3", {})
    if stage3:
        num_images = len(stage3.get("images", []))
        num_points = sum(p.numel() // 3 for p in stage3.get("pts3d", []))
        print(f"\nStage 3统计:")
        print(f"  重建图像数: {num_images}")
        print(f"  点云数量: {num_points:,}")


if __name__ == "__main__":
    main()
