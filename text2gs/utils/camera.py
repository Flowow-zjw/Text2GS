"""
Camera utilities
"""

import numpy as np
import torch
import cv2
from typing import Tuple, Optional


def get_intrinsic(fov: float, width: int, height: int) -> np.ndarray:
    """
    Compute camera intrinsic matrix
    
    Args:
        fov: Field of view in degrees
        width: Image width
        height: Image height
        
    Returns:
        3x3 intrinsic matrix K
    """
    f = 0.5 * width / np.tan(0.5 * fov / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float32)
    
    return K


def get_rotation(theta: float, phi: float) -> np.ndarray:
    """
    Compute rotation matrix from spherical angles
    
    Args:
        theta: Azimuth angle in degrees
        phi: Elevation angle in degrees
        
    Returns:
        3x3 rotation matrix R
    """
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = (R2 @ R1).astype(np.float32)
    
    return R


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world to world-to-camera"""
    return torch.linalg.inv(c2w)


def pose_to_pytorch3d(c2w: torch.Tensor, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pose to PyTorch3D camera format
    
    Args:
        c2w: (N, 4, 4) camera-to-world matrices
        device: Target device
        
    Returns:
        R: (N, 3, 3) rotation matrices (row-major)
        T: (N, 3) translation vectors
    """
    R, T = c2w[:, :3, :3], c2w[:, :3, 3:]
    
    # Convert from RDF to LUF coordinate system
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
    
    new_c2w = torch.cat([R, T], 2)
    n = c2w.shape[0]
    
    # Add homogeneous row
    bottom = torch.tensor([[[0, 0, 0, 1]]]).to(device).repeat(n, 1, 1)
    full_c2w = torch.cat((new_c2w, bottom), 1)
    
    # Invert to get w2c
    w2c = torch.linalg.inv(full_c2w)
    
    R_out = w2c[:, :3, :3].permute(0, 2, 1)  # Row-major
    T_out = w2c[:, :3, 3]
    
    return R_out, T_out


def interpolate_poses(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    num_steps: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Interpolate between two camera poses
    
    Args:
        start_pose: (4, 4) starting pose
        end_pose: (4, 4) ending pose
        num_steps: Number of interpolation steps
        device: Target device
        
    Returns:
        (num_steps, 4, 4) interpolated poses
    """
    from scipy.spatial.transform import Rotation, Slerp
    
    start_R = Rotation.from_matrix(start_pose[:3, :3].cpu().numpy())
    end_R = Rotation.from_matrix(end_pose[:3, :3].cpu().numpy())
    
    slerp = Slerp([0, 1], Rotation.from_quat([start_R.as_quat(), end_R.as_quat()]))
    
    poses = []
    for t in np.linspace(0, 1, num_steps):
        R = slerp(t).as_matrix()
        T = (1 - t) * start_pose[:3, 3].cpu().numpy() + t * end_pose[:3, 3].cpu().numpy()
        
        pose = torch.eye(4, device=device)
        pose[:3, :3] = torch.from_numpy(R).to(device)
        pose[:3, 3] = torch.from_numpy(T).to(device)
        poses.append(pose)
    
    return torch.stack(poses)
