"""
Stage 4: 3D Gaussian Splatting Training
"""

import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image
from scipy.spatial.transform import Rotation

from .base import BaseStage


class GaussianStage(BaseStage):
    """Export data and train 3D Gaussian Splatting"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        super().__init__(config, device)
        self.iterations = config.get("iterations", 7000)  # 默认 7000，可配置
        self.export_only = config.get("export_only", False)
        self.test_iterations = [self.iterations]
        self.save_iterations = [self.iterations]
        self.checkpoint_iterations = []
        self.gs_path = config.get("gaussian_splatting_path", "/root/autodl-tmp/gaussian-splatting")
        
        # 色彩优化参数
        self.sh_degree = config.get("sh_degree")  # 球谐函数阶数 (0-3)
        self.lambda_dssim = config.get("lambda_dssim")  # SSIM 损失权重
        
        # 几何优化参数
        self.opacity_reset_interval = config.get("opacity_reset_interval")  # 透明度重置间隔
        self.densify_grad_threshold = config.get("densify_grad_threshold")  # 密集化梯度阈值
        
        # 点云采样参数
        self.max_init_points = config.get("max_init_points", 500000)  # 初始点云最大数量（默认50万）
        
    def load_model(self) -> None:
        """3D-GS doesn't need pre-loading"""
        pass
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export data for 3D-GS training
        
        Args:
            inputs: dict with point cloud and views from Stage 3
            
        Returns:
            dict with export path and optionally trained model
        """
        import json
        
        output_dir = inputs.get("output_dir", "./output/3dgs")
        
        # Export COLMAP format
        export_info = self._export_colmap(inputs, output_dir)
        
        result = {
            "export_dir": output_dir,
            "colmap_dir": os.path.join(output_dir, "sparse/0"),
            "images_dir": os.path.join(output_dir, "images"),
            "num_images": export_info["num_images"],
            "num_points": export_info["num_points"],
        }
        
        # Save metadata
        metadata = {
            "export_dir": output_dir,
            "num_images": export_info["num_images"],
            "num_points": export_info["num_points"],
            "image_resolution": export_info["resolution"],
            "camera_model": "PINHOLE",
            "ready_for_training": True,
            "train_command": f"python train.py -s {output_dir} --iterations {self.iterations}"
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Optionally train 3D-GS
        if not self.export_only:
            print("\n[Training] Starting 3D Gaussian Splatting training...")
            model_result = self._train_3dgs(output_dir)
            if model_result:
                result["training"] = model_result
                result["trained_model_path"] = model_result["model_path"]
        else:
            print("\n[Export Only] Skipping training. Set 'export_only: false' to train.")
            print(f"To train manually, run:")
            print(f"  cd {self.gs_path}")
            print(f"  python train.py -s {output_dir} --iterations {self.iterations}")
        
        return result
    
    def _to_numpy(self, data):
        """Convert tensor/list to numpy array (replaces dust3r.utils.device.to_numpy)"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        elif hasattr(data, 'cpu'):
            return data.cpu().numpy()
        elif hasattr(data, 'numpy'):
            return data.numpy()
        else:
            return np.array(data)
    
    def _export_colmap(self, inputs: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Export data in COLMAP format using Stage 3 reconstructed images"""
        
        pts3d = inputs["pts3d"]
        imgs = inputs["images"]  # Stage 3 DUSt3R重建的24张图像
        c2ws = inputs["c2ws"]
        focals = inputs["focals"]
        principal_points = inputs["principal_points"]
        
        # 不再使用original_images和all_views
        # 直接使用Stage 3 DUSt3R重建的24张图像
        
        # Create directories
        images_dir = os.path.join(output_dir, "images")
        sparse_dir = os.path.join(output_dir, "sparse", "0")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        
        image_names = []
        image_poses = []
        image_camera_ids = []
        
        # Get focal length and principal point from DUSt3R
        if hasattr(focals, 'cpu'):
            focals_np = focals.cpu().numpy()
        else:
            focals_np = focals
        fx_dust3r = focals_np[0, 0] if len(focals_np.shape) > 1 else focals_np[0]
        
        if hasattr(principal_points, 'cpu'):
            pp_np = principal_points.cpu().numpy()
        else:
            pp_np = principal_points
        
        # DUSt3R image size (512x384)
        H_dust3r, W_dust3r = imgs[0].shape[:2]
        cx_dust3r = pp_np[0, 0] if len(pp_np.shape) > 1 else W_dust3r / 2
        cy_dust3r = pp_np[0, 1] if len(pp_np.shape) > 1 else H_dust3r / 2
        
        # 保存Stage 3重建的所有图像（24张）
        print(f"  Saving {len(imgs)} images from Stage 3 DUSt3R reconstruction...")
        for i, img in enumerate(imgs):
            name = f"image_{i:04d}.png"
            image_names.append(name)
            self._save_image(img, os.path.join(images_dir, name))
            image_camera_ids.append(1)  # 统一使用Camera 1
            
            # 使用Stage 3重建的位姿
            if hasattr(c2ws, 'cpu'):
                pose = c2ws[i].cpu().numpy()
            else:
                pose = c2ws[i]
            image_poses.append(pose)
        
        # Write cameras.txt - 使用统一的相机参数（基于DUSt3R的512x384）
        with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # Single camera model: All images are 512x384 from DUSt3R
            f.write(f"1 PINHOLE {W_dust3r} {H_dust3r} {fx_dust3r} {fx_dust3r} {cx_dust3r} {cy_dust3r}\n")
        
        # Write images.txt
        with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for i, (name, pose, cam_id) in enumerate(zip(image_names, image_poses, image_camera_ids)):
                c2w = pose
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3]
                t = w2c[:3, 3]
                
                quat = Rotation.from_matrix(R).as_quat()  # x, y, z, w
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n")
                f.write("\n")
        
        # Write points3D.txt
        pts3d_np = self._to_numpy(pts3d)
        
        if isinstance(imgs, list):
            imgs_np = [self._to_numpy(img) if not isinstance(img, np.ndarray) else img for img in imgs]
        else:
            imgs_np = imgs if isinstance(imgs, np.ndarray) else self._to_numpy(imgs)
        
        all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_np])
        all_cols = np.concatenate([p.reshape(-1, 3) for p in imgs_np])
        
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            
            step = max(1, len(all_pts) // self.max_init_points)
            num_saved_points = 0
            for i in range(0, len(all_pts), step):
                pt = all_pts[i]
                col = all_cols[i]
                r = int(col[0] * 255) if col[0] <= 1 else int(col[0])
                g = int(col[1] * 255) if col[1] <= 1 else int(col[1])
                b = int(col[2] * 255) if col[2] <= 1 else int(col[2])
                f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0\n")
                num_saved_points += 1
        
        print(f"  Exported {len(image_names)} images and {num_saved_points} points to {output_dir}")
        
        return {
            "num_images": len(image_names),
            "num_points": num_saved_points,
            "resolution": [H_dust3r, W_dust3r]
        }
    
    def _save_image(self, img: np.ndarray, path: str) -> None:
        """Save image array to file"""
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def _train_3dgs(self, data_dir: str):
        """Train 3D-GS using official gaussian-splatting"""
        import subprocess
        from datetime import datetime
        
        # Check if gaussian-splatting exists
        if not os.path.exists(self.gs_path):
            print(f"Warning: gaussian-splatting not found at {self.gs_path}")
            print("Please install it or update the path in the config")
            return None
        
        # Convert to absolute path
        data_dir = os.path.abspath(data_dir)
        
        # Prepare training command
        train_script = os.path.join(self.gs_path, "train.py")
        output_model_dir = os.path.join(data_dir, "output")
        log_dir = os.path.join(data_dir, "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        cmd = [
            "python", train_script,
            "-s", data_dir,
            "-m", output_model_dir,
            "--iterations", str(self.iterations),
            "--eval",  # Enable evaluation
        ]
        
        # Add test iterations
        if self.test_iterations:
            for it in self.test_iterations:
                cmd.extend(["--test_iterations", str(it)])
        
        # Add save iterations
        if self.save_iterations:
            for it in self.save_iterations:
                cmd.extend(["--save_iterations", str(it)])
        
        # Add checkpoint iterations
        if self.checkpoint_iterations:
            for it in self.checkpoint_iterations:
                cmd.extend(["--checkpoint_iterations", str(it)])
        
        # 色彩优化参数
        if self.sh_degree is not None:
            cmd.extend(["--sh_degree", str(self.sh_degree)])
        
        if self.lambda_dssim is not None:
            cmd.extend(["--lambda_dssim", str(self.lambda_dssim)])
        
        # 几何优化参数
        if self.opacity_reset_interval is not None:
            cmd.extend(["--opacity_reset_interval", str(self.opacity_reset_interval)])
        
        if self.densify_grad_threshold is not None:
            cmd.extend(["--densify_grad_threshold", str(self.densify_grad_threshold)])
        
        print(f"\n{'='*60}")
        print("Starting 3D Gaussian Splatting Training...")
        print(f"Source: {data_dir}")
        print(f"Output: {output_model_dir}")
        print(f"Iterations: {self.iterations}")
        if self.sh_degree is not None:
            print(f"SH Degree: {self.sh_degree}")
        if self.lambda_dssim is not None:
            print(f"Lambda DSSIM: {self.lambda_dssim}")
        if self.opacity_reset_interval is not None:
            print(f"Opacity Reset Interval: {self.opacity_reset_interval}")
        print(f"Log File: {log_file}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        # Save training configuration
        config_file = os.path.join(log_dir, f"training_config_{timestamp}.txt")
        with open(config_file, "w") as f:
            f.write("3D Gaussian Splatting Training Configuration\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Source Directory: {data_dir}\n")
            f.write(f"Output Directory: {output_model_dir}\n")
            f.write(f"Iterations: {self.iterations}\n")
            f.write(f"Test Iterations: {self.test_iterations}\n")
            f.write(f"Save Iterations: {self.save_iterations}\n")
            f.write(f"Checkpoint Iterations: {self.checkpoint_iterations}\n")
            f.write(f"\nCommand:\n{' '.join(cmd)}\n")
        
        try:
            # Run training and capture output
            with open(log_file, "w") as f:
                f.write(f"Training started at {timestamp}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("=" * 60 + "\n\n")
                f.flush()
                
                result = subprocess.run(
                    cmd,
                    cwd=self.gs_path,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            print(f"\n{'='*60}")
            print("Training completed successfully!")
            print(f"Model saved to: {output_model_dir}")
            print(f"Training log: {log_file}")
            print(f"{'='*60}\n")
            
            # Save completion status
            status_file = os.path.join(log_dir, "training_status.txt")
            with open(status_file, "w") as f:
                f.write("Training Status: SUCCESS\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {output_model_dir}\n")
                f.write(f"Iterations: {self.iterations}\n")
                f.write(f"Log File: {log_file}\n")
            
            return {
                "model_path": output_model_dir,
                "iterations": self.iterations,
                "success": True,
                "log_file": log_file,
                "config_file": config_file,
                "status_file": status_file,
            }
            
        except subprocess.CalledProcessError as e:
            print(f"\nError during training: {e}")
            
            # Save error status
            status_file = os.path.join(log_dir, "training_status.txt")
            with open(status_file, "w") as f:
                f.write("Training Status: FAILED\n")
                f.write(f"Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Log File: {log_file}\n")
            
            return None
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            
            # Save error status
            status_file = os.path.join(log_dir, "training_status.txt")
            with open(status_file, "w") as f:
                f.write("Training Status: ERROR\n")
                f.write(f"Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
            
            return None
