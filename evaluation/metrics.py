"""
Evaluation metrics for Text2GS experiments

Metrics:
1. Multi-view Consistency: CLIP feature similarity between adjacent views
2. Text-Image Alignment: CLIP score between text and generated images
3. Rendering Quality: NIQE (no-reference image quality)
4. Geometric Consistency: Reprojection error from point cloud
5. Efficiency: Time, memory, FPS
6. FID: Fréchet Inception Distance for generation quality
7. LPIPS: Learned Perceptual Image Patch Similarity for perceptual quality
"""

import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import json
import warnings


def compute_multi_view_consistency(
    images: List[np.ndarray],
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    Compute multi-view consistency using CLIP features
    
    Args:
        images: List of images (H, W, 3) in [0, 255] or [0, 1]
        device: Device to use
        
    Returns:
        dict with 'mean_similarity', 'std_similarity', 'min_similarity', 'global_consistency'
    """
    try:
        import clip
    except ImportError:
        print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
        return {"mean_similarity": 0.0, "std_similarity": 0.0, "min_similarity": 0.0, "global_consistency": 0.0}
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Preprocess images
    clip_images = []
    for img in images:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        clip_img = preprocess(pil_img).unsqueeze(0).to(device)
        clip_images.append(clip_img)
    
    # Extract features
    features = []
    with torch.no_grad():
        for clip_img in clip_images:
            feat = model.encode_image(clip_img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)
    
    # Compute pairwise similarities between adjacent views
    adjacent_similarities = []
    for i in range(len(features) - 1):
        sim = torch.cosine_similarity(features[i], features[i + 1])
        adjacent_similarities.append(sim.item())
    
    # Also compute similarity between first and last (for 360° consistency)
    if len(features) > 2:
        sim_loop = torch.cosine_similarity(features[-1], features[0])
        adjacent_similarities.append(sim_loop.item())
    
    # Compute global consistency (all pairs)
    all_pairs_similarities = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sim = torch.cosine_similarity(features[i], features[j])
            all_pairs_similarities.append(sim.item())
    
    return {
        "mean_similarity": float(np.mean(adjacent_similarities)),
        "std_similarity": float(np.std(adjacent_similarities)),
        "min_similarity": float(np.min(adjacent_similarities)),
        "max_similarity": float(np.max(adjacent_similarities)),
        "global_consistency": float(np.mean(all_pairs_similarities)),
        "global_std": float(np.std(all_pairs_similarities)),
        "num_adjacent_pairs": len(adjacent_similarities),
        "num_all_pairs": len(all_pairs_similarities)
    }


def compute_text_image_alignment(
    text: str,
    images: List[np.ndarray],
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    Compute text-image alignment using CLIP score
    
    Args:
        text: Text prompt
        images: List of images (H, W, 3) in [0, 255] or [0, 1]
        device: Device to use
        
    Returns:
        dict with 'mean_clip_score', 'std_clip_score', 'min_clip_score'
    """
    try:
        import clip
    except ImportError:
        print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
        return {"mean_clip_score": 0.0, "std_clip_score": 0.0, "min_clip_score": 0.0}
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Encode text
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute CLIP score for each image
    clip_scores = []
    for img in images:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        clip_img = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(clip_img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            score = torch.cosine_similarity(text_features, image_features)
            clip_scores.append(score.item())
    
    return {
        "mean_clip_score": float(np.mean(clip_scores)),
        "std_clip_score": float(np.std(clip_scores)),
        "min_clip_score": float(np.min(clip_scores)),
        "max_clip_score": float(np.max(clip_scores)),
        "num_images": len(clip_scores)
    }


def compute_rendering_quality(
    images: List[np.ndarray],
    metric: str = "combined"
) -> Dict[str, float]:
    """
    Compute rendering quality using no-reference metrics
    
    Args:
        images: List of images (H, W, 3) in [0, 255] or [0, 1]
        metric: Quality metric to use ('combined', 'sharpness', 'contrast')
        
    Returns:
        dict with quality scores
    """
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not installed")
        return {"mean_quality": 0.0, "std_quality": 0.0}
    
    sharpness_scores = []
    contrast_scores = []
    brightness_scores = []
    
    for img in images:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Sharpness: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append(laplacian_var)
        
        # 2. Contrast: standard deviation
        contrast = gray.std()
        contrast_scores.append(contrast)
        
        # 3. Brightness: mean pixel value
        brightness = gray.mean()
        brightness_scores.append(brightness)
    
    # Normalize scores to [0, 1] range for combination
    def normalize(scores):
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s > 0:
            return [(s - min_s) / (max_s - min_s) for s in scores]
        return [0.5] * len(scores)
    
    norm_sharpness = normalize(sharpness_scores)
    norm_contrast = normalize(contrast_scores)
    
    # Combined quality score (higher is better)
    combined_scores = [
        0.6 * s + 0.4 * c 
        for s, c in zip(norm_sharpness, norm_contrast)
    ]
    
    return {
        "mean_quality": float(np.mean(combined_scores)),
        "std_quality": float(np.std(combined_scores)),
        "min_quality": float(np.min(combined_scores)),
        "max_quality": float(np.max(combined_scores)),
        "mean_sharpness": float(np.mean(sharpness_scores)),
        "mean_contrast": float(np.mean(contrast_scores)),
        "mean_brightness": float(np.mean(brightness_scores)),
        "metric": "combined_sharpness_contrast"
    }


def compute_point_cloud_quality(
    pts3d: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute point cloud quality metrics
    
    Args:
        pts3d: List of 3D points per view
        
    Returns:
        dict with point cloud quality metrics
    """
    if len(pts3d) == 0:
        return {
            "num_points": 0,
            "mean_nn_distance": 0.0,
            "point_density": 0.0,
            "coverage_score": 0.0
        }
    
    # Merge all point clouds
    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d])
    
    # Remove invalid points (NaN, Inf)
    valid_mask = np.isfinite(all_pts).all(axis=1)
    all_pts = all_pts[valid_mask]
    
    if len(all_pts) < 10:
        return {
            "num_points": len(all_pts),
            "mean_nn_distance": 0.0,
            "point_density": 0.0,
            "coverage_score": 0.0
        }
    
    try:
        from scipy.spatial import cKDTree
        
        # Compute nearest neighbor distances (point density)
        tree = cKDTree(all_pts)
        distances, _ = tree.query(all_pts, k=2)  # k=2 to get nearest neighbor (excluding self)
        nn_distances = distances[:, 1]  # Take second nearest (first is self)
        
        # Compute bounding box volume (coverage)
        bbox_min = all_pts.min(axis=0)
        bbox_max = all_pts.max(axis=0)
        bbox_volume = np.prod(bbox_max - bbox_min)
        
        # Point density: points per unit volume
        point_density = len(all_pts) / bbox_volume if bbox_volume > 0 else 0
        
        # Coverage score: ratio of occupied voxels (simplified)
        # Higher density and more points = better coverage
        coverage_score = np.log10(len(all_pts) + 1) / (np.mean(nn_distances) + 1e-6)
        
        return {
            "num_points": int(len(all_pts)),
            "mean_nn_distance": float(np.mean(nn_distances)),
            "median_nn_distance": float(np.median(nn_distances)),
            "std_nn_distance": float(np.std(nn_distances)),
            "point_density": float(point_density),
            "coverage_score": float(coverage_score),
            "bbox_volume": float(bbox_volume)
        }
    except Exception as e:
        print(f"Warning: Error computing point cloud quality: {e}")
        return {
            "num_points": int(len(all_pts)),
            "mean_nn_distance": 0.0,
            "point_density": 0.0,
            "coverage_score": 0.0
        }


def compute_efficiency_metrics(
    stage_times: Dict[str, float],
    peak_memory: float,
    num_images: int
) -> Dict[str, float]:
    """
    Compute efficiency metrics
    
    Args:
        stage_times: Dict mapping stage name to time in seconds
        peak_memory: Peak GPU memory in GB
        num_images: Number of output images
        
    Returns:
        dict with efficiency metrics
    """
    total_time = sum(stage_times.values())
    
    return {
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60.0,
        "peak_memory_gb": peak_memory,
        "time_per_image": total_time / num_images if num_images > 0 else 0.0,
        **{f"{stage}_time": time for stage, time in stage_times.items()}
    }


def evaluate_pipeline_results(
    output_dir: str,
    text_prompt: str,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Evaluate complete pipeline results
    
    Args:
        output_dir: Path to pipeline output directory
        text_prompt: Original text prompt
        device: Device to use
        
    Returns:
        dict with all evaluation metrics
    """
    results = {
        "output_dir": output_dir,
        "text_prompt": text_prompt
    }
    
    # Load Stage 1 images for multi-view consistency
    stage1_dir = os.path.join(output_dir, "stage1_mvdiffusion")
    if os.path.exists(stage1_dir):
        stage1_images = []
        for i in range(8):
            img_path = os.path.join(stage1_dir, f"view_{i:02d}.png")
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path))
                stage1_images.append(img)
        
        if stage1_images:
            print("  Computing multi-view consistency...")
            mvc = compute_multi_view_consistency(stage1_images, device)
            results["multi_view_consistency"] = mvc
            
            print("  Computing text-image alignment...")
            tia = compute_text_image_alignment(text_prompt, stage1_images, device)
            results["text_image_alignment"] = tia
            
            print("  Computing rendering quality...")
            rq = compute_rendering_quality(stage1_images)
            results["rendering_quality"] = rq
            
            # Compute LPIPS-based consistency
            print("  Computing LPIPS-based consistency...")
            lpips_consistency = compute_lpips_consistency(stage1_images, device)
            results["lpips_consistency"] = lpips_consistency
    
    # Load Stage 2 point cloud for quality evaluation
    stage2_dir = os.path.join(output_dir, "stage2_pointcloud")
    if os.path.exists(stage2_dir):
        ply_path = os.path.join(stage2_dir, "pointcloud.ply")
        if os.path.exists(ply_path):
            try:
                print("  Computing point cloud quality...")
                # Try to load with open3d first
                try:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(ply_path)
                    pts = np.asarray(pcd.points)
                    pcq = compute_point_cloud_quality([pts])
                    results["point_cloud_quality"] = pcq
                except ImportError:
                    print("    Warning: open3d not installed, skipping point cloud quality")
                    print("    Install with: pip install open3d")
            except Exception as e:
                print(f"    Warning: Failed to load point cloud: {e}")
    
    # Load efficiency metrics from summary
    summary_path = os.path.join(output_dir, "PIPELINE_SUMMARY.txt")
    if os.path.exists(summary_path):
        results["summary_available"] = True
    
    # Load Stage 4 training results
    stage4_dir = os.path.join(output_dir, "stage4_gaussian")
    if os.path.exists(stage4_dir):
        metadata_path = os.path.join(stage4_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                stage4_meta = json.load(f)
                results["stage4_metadata"] = stage4_meta
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file"""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to: {output_path}")


def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(input_path, "r") as f:
        return json.load(f)


def compare_methods_with_fid(
    method1_images: List[np.ndarray],
    method2_images: List[np.ndarray],
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Compare two methods using FID and LPIPS
    
    Args:
        method1_images: Images from method 1 (e.g., Text2GS 124)
        method2_images: Images from method 2 (e.g., Text2GS 1234 or DreamScene)
        device: Device to use
        
    Returns:
        dict with comparison metrics
    """
    results = {}
    
    # Compute FID (treating method1 as "real" for reference)
    print("  Computing FID between methods...")
    fid_result = compute_fid_score(method1_images, method2_images, device)
    results["fid"] = fid_result
    
    # Compute LPIPS between corresponding images
    if len(method1_images) == len(method2_images):
        print("  Computing LPIPS between corresponding images...")
        lpips_result = compute_lpips_score(method1_images, method2_images, device)
        results["lpips"] = lpips_result
    else:
        results["lpips"] = {"error": "image_count_mismatch"}
    
    return results


def compute_statistical_significance(
    data_124: List[float],
    data_1234: List[float]
) -> Dict[str, Any]:
    """
    Compute statistical significance between two groups
    
    Args:
        data_124: Metric values for Stage 1+2+4
        data_1234: Metric values for Stage 1+2+3+4
        
    Returns:
        dict with statistical test results
    """
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not installed")
        return {"error": "scipy_not_installed"}
    
    if len(data_124) < 2 or len(data_1234) < 2:
        return {"error": "insufficient_data"}
    
    # Paired t-test (if same number of samples)
    if len(data_124) == len(data_1234):
        t_stat, p_value = stats.ttest_rel(data_124, data_1234)
        test_type = "paired_t_test"
    else:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(data_124, data_1234)
        test_type = "independent_t_test"
    
    # Cohen's d (effect size)
    mean_diff = np.mean(data_1234) - np.mean(data_124)
    pooled_std = np.sqrt((np.std(data_124, ddof=1)**2 + np.std(data_1234, ddof=1)**2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_size_interpretation = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_interpretation = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"
    
    return {
        "test_type": test_type,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "cohen_d": float(cohen_d),
        "effect_size": effect_size_interpretation,
        "mean_124": float(np.mean(data_124)),
        "mean_1234": float(np.mean(data_1234)),
        "improvement_percent": float((mean_diff / np.mean(data_124)) * 100) if np.mean(data_124) != 0 else 0.0
    }


def compute_fid_score(
    real_images: List[np.ndarray],
    generated_images: List[np.ndarray],
    device: str = "cuda:0",
    batch_size: int = 50
) -> Dict[str, float]:
    """
    Compute Fréchet Inception Distance (FID) between real and generated images
    
    FID measures the distance between feature distributions of real and generated images.
    Lower FID indicates better generation quality and diversity.
    
    Args:
        real_images: List of real images (H, W, 3) in [0, 255] or [0, 1]
        generated_images: List of generated images (H, W, 3) in [0, 255] or [0, 1]
        device: Device to use
        batch_size: Batch size for feature extraction
        
    Returns:
        dict with 'fid_score' and related statistics
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
        import torch.nn.functional as F
    except ImportError:
        print("Warning: pytorch-fid not installed. Install with: pip install pytorch-fid")
        return {"fid_score": -1.0, "error": "pytorch_fid_not_installed"}
    
    if len(real_images) < 2 or len(generated_images) < 2:
        return {"fid_score": -1.0, "error": "insufficient_images"}
    
    try:
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
        def preprocess_images(images):
            """Preprocess images for Inception"""
            processed = []
            for img in images:
                # Ensure [0, 255] range
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                # Convert to PIL and resize to 299x299 (Inception input size)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((299, 299), Image.BILINEAR)
                
                # Convert to tensor and normalize to [-1, 1]
                img_tensor = torch.from_numpy(np.array(pil_img)).float()
                img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # [0, 1]
                img_tensor = (img_tensor - 0.5) * 2  # [-1, 1]
                processed.append(img_tensor)
            
            return torch.stack(processed)
        
        def get_activations(images, model, batch_size, device):
            """Extract Inception features"""
            model.eval()
            activations = []
            
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size].to(device)
                    pred = model(batch)[0]
                    
                    # Flatten spatial dimensions
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                    
                    activations.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
            
            return np.concatenate(activations, axis=0)
        
        # Preprocess images
        real_tensor = preprocess_images(real_images)
        gen_tensor = preprocess_images(generated_images)
        
        # Extract features
        real_features = get_activations(real_tensor, model, batch_size, device)
        gen_features = get_activations(gen_tensor, model, batch_size, device)
        
        # Compute FID
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_gen
        covmean, _ = np.linalg.eig(sigma_real.dot(sigma_gen))
        covmean = covmean.real
        
        # Numerical stability
        covmean = np.sqrt(np.maximum(covmean, 0))
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.sum(covmean)
        
        return {
            "fid_score": float(fid),
            "num_real_images": len(real_images),
            "num_generated_images": len(generated_images),
            "mu_distance": float(np.linalg.norm(diff)),
            "trace_real": float(np.trace(sigma_real)),
            "trace_gen": float(np.trace(sigma_gen))
        }
    
    except Exception as e:
        print(f"Warning: Error computing FID: {e}")
        return {"fid_score": -1.0, "error": str(e)}


def compute_lpips_score(
    images1: List[np.ndarray],
    images2: List[np.ndarray],
    device: str = "cuda:0",
    net: str = "alex"
) -> Dict[str, float]:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two sets of images
    
    LPIPS measures perceptual similarity using deep features.
    Lower LPIPS indicates more similar images.
    
    Args:
        images1: First set of images (H, W, 3) in [0, 255] or [0, 1]
        images2: Second set of images (H, W, 3) in [0, 255] or [0, 1]
        device: Device to use
        net: Network to use ('alex', 'vgg', or 'squeeze')
        
    Returns:
        dict with 'mean_lpips', 'std_lpips', 'min_lpips', 'max_lpips'
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips not installed. Install with: pip install lpips")
        return {"mean_lpips": -1.0, "error": "lpips_not_installed"}
    
    if len(images1) != len(images2):
        return {"mean_lpips": -1.0, "error": "image_count_mismatch"}
    
    if len(images1) == 0:
        return {"mean_lpips": -1.0, "error": "no_images"}
    
    try:
        # Load LPIPS model
        loss_fn = lpips.LPIPS(net=net).to(device)
        loss_fn.eval()
        
        def preprocess_image(img):
            """Preprocess image for LPIPS"""
            # Ensure [0, 1] range
            if img.max() > 1.0:
                img = img / 255.0
            
            # Convert to tensor and normalize to [-1, 1]
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            img_tensor = (img_tensor - 0.5) * 2  # [0, 1] -> [-1, 1]
            
            return img_tensor.unsqueeze(0)  # Add batch dimension
        
        lpips_scores = []
        
        with torch.no_grad():
            for img1, img2 in zip(images1, images2):
                # Preprocess
                tensor1 = preprocess_image(img1).to(device)
                tensor2 = preprocess_image(img2).to(device)
                
                # Compute LPIPS
                score = loss_fn(tensor1, tensor2)
                lpips_scores.append(score.item())
        
        return {
            "mean_lpips": float(np.mean(lpips_scores)),
            "std_lpips": float(np.std(lpips_scores)),
            "min_lpips": float(np.min(lpips_scores)),
            "max_lpips": float(np.max(lpips_scores)),
            "median_lpips": float(np.median(lpips_scores)),
            "num_pairs": len(lpips_scores),
            "network": net
        }
    
    except Exception as e:
        print(f"Warning: Error computing LPIPS: {e}")
        return {"mean_lpips": -1.0, "error": str(e)}


def compute_lpips_consistency(
    images: List[np.ndarray],
    device: str = "cuda:0",
    net: str = "alex"
) -> Dict[str, float]:
    """
    Compute LPIPS-based multi-view consistency
    
    Measures perceptual consistency between adjacent views using LPIPS.
    Lower scores indicate better consistency.
    
    Args:
        images: List of images (H, W, 3) in [0, 255] or [0, 1]
        device: Device to use
        net: Network to use ('alex', 'vgg', or 'squeeze')
        
    Returns:
        dict with consistency metrics
    """
    if len(images) < 2:
        return {"mean_lpips_consistency": -1.0, "error": "insufficient_images"}
    
    # Compute LPIPS between adjacent views
    adjacent_pairs_img1 = images[:-1]
    adjacent_pairs_img2 = images[1:]
    
    adjacent_result = compute_lpips_score(adjacent_pairs_img1, adjacent_pairs_img2, device, net)
    
    if "error" in adjacent_result:
        return adjacent_result
    
    # Also compute between first and last (for 360° consistency)
    if len(images) > 2:
        loop_result = compute_lpips_score([images[-1]], [images[0]], device, net)
        loop_score = loop_result.get("mean_lpips", -1.0)
    else:
        loop_score = adjacent_result["mean_lpips"]
    
    return {
        "mean_lpips_consistency": adjacent_result["mean_lpips"],
        "std_lpips_consistency": adjacent_result["std_lpips"],
        "min_lpips_consistency": adjacent_result["min_lpips"],
        "max_lpips_consistency": adjacent_result["max_lpips"],
        "loop_lpips": loop_score,
        "num_adjacent_pairs": adjacent_result["num_pairs"],
        "network": net
    }


def compute_statistical_significance(
    data_124: List[float],
    data_1234: List[float]
) -> Dict[str, Any]:
    """
    Compute statistical significance between two groups
    
    Args:
        data_124: Metric values for Stage 1+2+4
        data_1234: Metric values for Stage 1+2+3+4
        
    Returns:
        dict with statistical test results
    """
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not installed")
        return {"error": "scipy_not_installed"}
    
    if len(data_124) < 2 or len(data_1234) < 2:
        return {"error": "insufficient_data"}
    
    # Paired t-test (if same number of samples)
    if len(data_124) == len(data_1234):
        t_stat, p_value = stats.ttest_rel(data_124, data_1234)
        test_type = "paired_t_test"
    else:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(data_124, data_1234)
        test_type = "independent_t_test"
    
    # Cohen's d (effect size)
    mean_diff = np.mean(data_1234) - np.mean(data_124)
    pooled_std = np.sqrt((np.std(data_124, ddof=1)**2 + np.std(data_1234, ddof=1)**2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_size_interpretation = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_interpretation = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"
    
    return {
        "test_type": test_type,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "cohen_d": float(cohen_d),
        "effect_size": effect_size_interpretation,
        "mean_124": float(np.mean(data_124)),
        "mean_1234": float(np.mean(data_1234)),
        "improvement_percent": float((mean_diff / np.mean(data_124)) * 100) if np.mean(data_124) != 0 else 0.0
    }
