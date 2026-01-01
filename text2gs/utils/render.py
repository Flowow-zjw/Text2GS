"""
Rendering utilities
"""

import torch
import numpy as np
from typing import Tuple, Optional


def setup_point_renderer(
    cameras,
    image_size: Tuple[int, int],
    radius: float = 0.01,
    points_per_pixel: int = 10
):
    """
    Setup PyTorch3D point cloud renderer
    
    Args:
        cameras: PyTorch3D cameras
        image_size: (H, W) output image size
        radius: Point radius
        points_per_pixel: Points per pixel for rasterization
        
    Returns:
        PointsRenderer instance
    """
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
    )
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    
    return renderer


def render_pointcloud(
    points: torch.Tensor,
    colors: torch.Tensor,
    cameras,
    image_size: Tuple[int, int],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Render point cloud from given cameras
    
    Args:
        points: (N, 3) point positions
        colors: (N, 3) point colors
        cameras: PyTorch3D cameras
        image_size: (H, W) output size
        device: Target device
        
    Returns:
        (num_views, H, W, 4) rendered images with alpha
    """
    from pytorch3d.structures import Pointclouds
    
    num_views = len(cameras)
    
    point_cloud = Pointclouds(
        points=[points.to(device)],
        features=[colors.to(device)]
    ).extend(num_views)
    
    renderer = setup_point_renderer(cameras, image_size)
    images = renderer(point_cloud)
    
    return images
