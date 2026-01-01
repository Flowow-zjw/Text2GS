"""
I/O utilities
"""

import os
import torch
import numpy as np
import torchvision
from PIL import Image
from typing import Union, List, Optional
import trimesh


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: str
) -> None:
    """Save image to file"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    Image.fromarray(image).save(path)


def save_video(
    frames: Union[np.ndarray, torch.Tensor],
    path: str,
    fps: int = 8
) -> None:
    """Save frames as video"""
    if isinstance(frames, np.ndarray):
        tensor_data = (torch.from_numpy(frames) * 255).to(torch.uint8)
    elif isinstance(frames, torch.Tensor):
        tensor_data = (frames.detach().cpu() * 255).to(torch.uint8)
    
    torchvision.io.write_video(
        path, tensor_data, fps=fps,
        video_codec="h264", options={"crf": "10"}
    )


def save_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    path: str,
    normals: Optional[np.ndarray] = None
) -> None:
    """
    Save point cloud as PLY file
    
    Args:
        points: (N, 3) point positions
        colors: (N, 3) point colors (0-1 or 0-255)
        path: Output path
        normals: Optional (N, 3) normals
    """
    if colors.max() <= 1:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    pcd = trimesh.PointCloud(points, colors=colors)
    
    if normals is not None:
        pcd.vertices_normal = normals
    
    pcd.export(path)


def load_images_as_tensor(
    paths: List[str],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load images as tensor
    
    Args:
        paths: List of image paths
        device: Target device
        
    Returns:
        (N, H, W, 3) tensor with values in [0, 1]
    """
    images = []
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        images.append(img)
    
    images = np.stack(images)
    tensor = torch.from_numpy(images).float() / 255.0
    
    return tensor.to(device)
