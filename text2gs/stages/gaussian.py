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
        self.iterations = config.get("iterations", 2000)
        self.export_only = config.get("export_only", True)
        
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
            try:
                model = self._train_3dgs(output_dir)
                result["model"] = model
            except ImportError:
                print("3D-GS package not found. Please train manually.")
        
        return result
    
    def _export_colmap(self, inputs: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Export data in COLMAP format with both original and generated images"""
        from dust3r.utils.device import to_numpy
        
        pts3d = inputs["pts3d"]
        imgs = inputs["images"]  # DUSt3R processed images
        c2ws = inputs["c2ws"]
        focals = inputs["focals"]
        principal_points = inputs["principal_points"]
        all_views = inputs["all_views"]
        original_images = inputs.get("original_images")  # MVDiffusion original 512x512
        
        # Get interpolated camera poses if available
        c2ws_interp = inputs.get("c2ws_interp")
        num_input_views = inputs.get("num_input_views", len(imgs))
        video_length = inputs.get("video_length", 25)
        
        # Create directories
        sparse_dir = os.path.join(output_dir, "sparse", "0")
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        image_names = []
        image_poses = []
        image_camera_ids = []  # Track which camera each image uses
        
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
        
        # Save original images from MVDiffusion (512x512) - Camera 1
        if original_images is not None:
            H_orig, W_orig = original_images[0].shape[:2]
            # Scale focal length for original resolution
            fx_orig = fx_dust3r * W_orig / W_dust3r
            cx_orig = cx_dust3r * W_orig / W_dust3r
            cy_orig = cy_dust3r * H_orig / H_dust3r
            
            for i, img in enumerate(original_images):
                name = f"orig_{i:04d}.png"
                image_names.append(name)
                self._save_image(img, os.path.join(images_dir, name))
                image_camera_ids.append(1)  # Camera 1 for original
                
                # Use original pose
                if hasattr(c2ws, 'cpu'):
                    pose = c2ws[i].cpu().numpy()
                else:
                    pose = c2ws[i]
                image_poses.append(pose)
        else:
            H_orig, W_orig = 512, 512
            fx_orig, cx_orig, cy_orig = fx_dust3r, cx_dust3r, cy_dust3r
        
        # Save generated views from ViewCrafter (576x1024) - Camera 2
        generated_views = all_views[0] if len(all_views) > 0 else None
        
        if generated_views is not None and c2ws_interp is not None:
            H_gen, W_gen = generated_views.shape[1], generated_views.shape[2]
            # Scale focal length for generated resolution
            fx_gen = fx_dust3r * W_gen / W_dust3r
            cx_gen = cx_dust3r * W_gen / W_dust3r
            cy_gen = cy_dust3r * H_gen / H_dust3r
            
            if hasattr(c2ws_interp, 'cpu'):
                c2ws_interp_np = c2ws_interp.cpu().numpy()
            else:
                c2ws_interp_np = c2ws_interp
            
            for frame_idx in range(generated_views.shape[0]):
                frame = generated_views[frame_idx]
                if hasattr(frame, 'cpu'):
                    frame_np = ((frame.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                else:
                    frame_np = ((frame + 1) / 2 * 255).astype(np.uint8)
                
                name = f"gen_{frame_idx:04d}.png"
                image_names.append(name)
                Image.fromarray(frame_np).save(os.path.join(images_dir, name))
                image_camera_ids.append(2)  # Camera 2 for generated
                
                if frame_idx < len(c2ws_interp_np):
                    image_poses.append(c2ws_interp_np[frame_idx])
                else:
                    image_poses.append(c2ws_interp_np[-1])
        else:
            H_gen, W_gen = H_dust3r, W_dust3r
            fx_gen, cx_gen, cy_gen = fx_dust3r, cx_dust3r, cy_dust3r
        
        # Write cameras.txt - two cameras with different resolutions
        with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # Camera 1: Original images (512x512)
            f.write(f"1 PINHOLE {W_orig} {H_orig} {fx_orig} {fx_orig} {cx_orig} {cy_orig}\n")
            # Camera 2: Generated images (576x1024)
            f.write(f"2 PINHOLE {W_gen} {H_gen} {fx_gen} {fx_gen} {cx_gen} {cy_gen}\n")
        
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
        pts3d_np = to_numpy(pts3d)
        
        if isinstance(imgs, list):
            imgs_np = [to_numpy(img) if not isinstance(img, np.ndarray) else img for img in imgs]
        else:
            imgs_np = imgs if isinstance(imgs, np.ndarray) else to_numpy(imgs)
        
        all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_np])
        all_cols = np.concatenate([p.reshape(-1, 3) for p in imgs_np])
        
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            
            step = max(1, len(all_pts) // 100000)
            num_saved_points = 0
            for i in range(0, len(all_pts), step):
                pt = all_pts[i]
                col = all_cols[i]
                r = int(col[0] * 255) if col[0] <= 1 else int(col[0])
                g = int(col[1] * 255) if col[1] <= 1 else int(col[1])
                b = int(col[2] * 255) if col[2] <= 1 else int(col[2])
                f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0\n")
                num_saved_points += 1
        
        print(f"  Exported {len(image_names)} images ({num_input_views} orig + {len(image_names) - num_input_views} gen) and {num_saved_points} points to {output_dir}")
        
        return {
            "num_images": len(image_names),
            "num_points": num_saved_points,
            "resolution": [H_gen, W_gen]
        }
    
    def _save_image(self, img: np.ndarray, path: str) -> None:
        """Save image array to file"""
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def _train_3dgs(self, data_dir: str):
        """Train 3D-GS (requires gaussian-splatting package)"""
        # This would integrate with official gaussian-splatting
        # For now, return None
        return None
