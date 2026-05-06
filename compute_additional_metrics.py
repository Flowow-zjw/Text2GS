#!/usr/bin/env python3
"""
计算额外的评估指标（FID、点云质量、统计显著性）
使用已保存的实验结果，不需要重新运行实验

用法：
python compute_additional_metrics.py --experiment-dir /path/to/progressive_comparison_TIMESTAMP
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any

# 导入评估函数
from evaluation.metrics import (
    compute_fid_score,
    compute_point_cloud_quality,
    compute_statistical_significance
)


def parse_args():
    parser = argparse.ArgumentParser(description="计算额外的评估指标")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="实验结果目录路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备"
    )
    parser.add_argument(
        "--compute-fid",
        action="store_true",
        help="计算FID (124 vs 1234)"
    )
    parser.add_argument(
        "--compute-pointcloud",
        action="store_true",
        help="计算点云质量"
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="计算统计显著性"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="计算所有额外指标"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示目录结构调试信息"
    )
    return parser.parse_args()


def load_images_from_dir(image_dir: str) -> List[np.ndarray]:
    """从目录加载所有图像"""
    images = []
    image_files = sorted(Path(image_dir).glob("*.png"))
    
    for img_path in image_files:
        img = np.array(Image.open(img_path))
        images.append(img)
    
    return images


def load_point_cloud(ply_path: str) -> np.ndarray:
    """加载点云文件"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        return np.asarray(pcd.points)
    except ImportError:
        print("Warning: open3d not installed. Install with: pip install open3d")
        return None
    except Exception as e:
        print(f"Warning: Failed to load point cloud: {e}")
        return None


def compute_fid_between_variants(
    experiment_dir: str,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    计算Stage 124和1234之间的FID
    将124作为"真实"图像，1234作为"生成"图像
    """
    print("\n" + "=" * 80)
    print("计算FID (Stage 124 vs Stage 1234)")
    print("=" * 80)
    
    results = {
        "per_prompt": [],
        "overall": {}
    }
    
    # 遍历所有提示词
    prompt_dirs = sorted(Path(experiment_dir).glob("prompt_*"))
    
    all_images_124 = []
    all_images_1234 = []
    
    for prompt_dir in prompt_dirs:
        prompt_name = prompt_dir.name
        print(f"\n处理 {prompt_name}...")
        
        # 查找124和1234的图像目录
        # 使用stage1_mvdiffusion（与原始评估一致）
        variant_124_dirs = list(prompt_dir.glob("variant_124/run_*/*/stage1_mvdiffusion"))
        variant_1234_dirs = list(prompt_dir.glob("variant_1234/run_*/*/stage1_mvdiffusion"))
        
        # Fallback: 尝试不带run_前缀的路径
        if not variant_124_dirs:
            variant_124_dirs = list(prompt_dir.glob("variant_124/*/stage1_mvdiffusion"))
        if not variant_1234_dirs:
            variant_1234_dirs = list(prompt_dir.glob("variant_1234/*/stage1_mvdiffusion"))
        
        # 调试信息
        if not variant_124_dirs:
            print(f"  警告: 未找到124图像")
            print(f"    尝试的路径: {prompt_dir}/variant_124/run_*/*/stage1_mvdiffusion")
        
        if not variant_1234_dirs:
            print(f"  警告: 未找到1234图像")
            print(f"    尝试的路径: {prompt_dir}/variant_1234/run_*/*/stage1_mvdiffusion")
        
        if not variant_124_dirs or not variant_1234_dirs:
            print(f"  跳过: 缺少图像目录")
            continue
        
        # 加载图像
        images_124 = load_images_from_dir(str(variant_124_dirs[0]))
        images_1234 = load_images_from_dir(str(variant_1234_dirs[0]))
        
        if not images_124 or not images_1234:
            print(f"  跳过: 图像加载失败")
            continue
        
        print(f"  加载了 {len(images_124)} 张124图像, {len(images_1234)} 张1234图像")
        
        # 累积所有图像
        all_images_124.extend(images_124)
        all_images_1234.extend(images_1234)
        
        # 计算单个提示词的FID
        fid_result = compute_fid_score(images_124, images_1234, device)
        results["per_prompt"].append({
            "prompt": prompt_name,
            "fid": fid_result
        })
        
        if "error" not in fid_result:
            print(f"  FID: {fid_result['fid_score']:.4f}")
    
    # 计算总体FID
    if all_images_124 and all_images_1234:
        print(f"\n计算总体FID...")
        print(f"  总共: {len(all_images_124)} 张124图像, {len(all_images_1234)} 张1234图像")
        
        overall_fid = compute_fid_score(all_images_124, all_images_1234, device)
        results["overall"] = overall_fid
        
        if "error" not in overall_fid:
            print(f"  总体FID: {overall_fid['fid_score']:.4f}")
    
    return results


def compute_pointcloud_quality_all(
    experiment_dir: str
) -> Dict[str, Any]:
    """计算所有提示词的点云质量"""
    print("\n" + "=" * 80)
    print("计算点云质量")
    print("=" * 80)
    
    results = {
        "124": {"per_prompt": [], "overall": {}},
        "1234": {"per_prompt": [], "overall": {}}
    }
    
    # 遍历所有提示词
    prompt_dirs = sorted(Path(experiment_dir).glob("prompt_*"))
    
    for variant in ["124", "1234"]:
        print(f"\n处理 Stage {variant}...")
        
        all_points = []
        
        for prompt_dir in prompt_dirs:
            prompt_name = prompt_dir.name
            
            # 查找点云文件
            # 实际路径: variant_124/run_0/TIMESTAMP/stage2_pointcloud/pointcloud.ply
            ply_files = list(prompt_dir.glob(f"variant_{variant}/run_*/*/stage2_pointcloud/pointcloud.ply"))
            if not ply_files:
                ply_files = list(prompt_dir.glob(f"variant_{variant}/*/stage2_pointcloud/pointcloud.ply"))
            if not ply_files:
                ply_files = list(prompt_dir.glob(f"variant_{variant}/run_*/stage2_pointcloud/pointcloud.ply"))
            
            if not ply_files:
                continue
            
            # 加载点云
            points = load_point_cloud(str(ply_files[0]))
            
            if points is None or len(points) == 0:
                continue
            
            print(f"  {prompt_name}: {len(points)} 个点")
            
            # 计算单个提示词的点云质量
            pcq = compute_point_cloud_quality([points])
            results[variant]["per_prompt"].append({
                "prompt": prompt_name,
                "quality": pcq
            })
            
            all_points.append(points)
        
        # 计算总体点云质量
        if all_points:
            print(f"\n  计算总体点云质量...")
            overall_pcq = compute_point_cloud_quality(all_points)
            results[variant]["overall"] = overall_pcq
            
            print(f"  总点数: {overall_pcq.get('num_points', 0)}")
            print(f"  平均最近邻距离: {overall_pcq.get('mean_nn_distance', 0):.6f}")
            print(f"  点密度: {overall_pcq.get('point_density', 0):.2f}")
    
    return results


def compute_stats_from_results(
    results_json_path: str
) -> Dict[str, Any]:
    """从results.json计算统计显著性"""
    print("\n" + "=" * 80)
    print("计算统计显著性检验")
    print("=" * 80)
    
    # 读取结果
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # 收集数据
    metrics_data = {
        '124': {
            'clip_mvc': [],
            'clip_text': [],
            'rendering_quality': [],
            'lpips': []
        },
        '1234': {
            'clip_mvc': [],
            'clip_text': [],
            'rendering_quality': [],
            'lpips': []
        }
    }
    
    for result in results:
        for variant in ['124', '1234']:
            if variant in result['variants']:
                v = result['variants'][variant]
                if 'evaluation' in v:
                    eval_data = v['evaluation']
                    
                    if 'multi_view_consistency' in eval_data:
                        metrics_data[variant]['clip_mvc'].append(
                            eval_data['multi_view_consistency']['mean_similarity']
                        )
                    
                    if 'text_image_alignment' in eval_data:
                        metrics_data[variant]['clip_text'].append(
                            eval_data['text_image_alignment']['mean_clip_score']
                        )
                    
                    if 'rendering_quality' in eval_data:
                        metrics_data[variant]['rendering_quality'].append(
                            eval_data['rendering_quality']['mean_quality']
                        )
                    
                    if 'lpips_consistency' in eval_data:
                        metrics_data[variant]['lpips'].append(
                            eval_data['lpips_consistency']['mean_lpips_consistency']
                        )
    
    # 进行统计检验
    stats_results = {}
    
    metrics_to_test = [
        ('clip_mvc', 'CLIP 多视角一致性'),
        ('clip_text', 'CLIP 文本对齐'),
        ('rendering_quality', '渲染质量'),
        ('lpips', 'LPIPS 一致性')
    ]
    
    print(f"\n样本量: {len(metrics_data['124']['clip_mvc'])} 个提示词\n")
    
    for metric_key, metric_name in metrics_to_test:
        data_124 = metrics_data['124'][metric_key]
        data_1234 = metrics_data['1234'][metric_key]
        
        if not data_124 or not data_1234:
            continue
        
        print(f"\n{metric_name}:")
        
        stats = compute_statistical_significance(data_124, data_1234)
        
        if 'error' not in stats:
            print(f"  p值: {stats['p_value']:.6f}")
            print(f"  Cohen's d: {stats['cohen_d']:.4f}")
            print(f"  效应量: {stats['effect_size']}")
            print(f"  显著性(α=0.05): {'是' if stats['significant_at_0.05'] else '否'}")
            
            stats_results[metric_key] = stats
    
    return stats_results


def main():
    args = parse_args()
    
    # 检查实验目录
    if not os.path.exists(args.experiment_dir):
        print(f"错误: 实验目录不存在: {args.experiment_dir}")
        return
    
    print("=" * 80)
    print("计算额外的评估指标")
    print("=" * 80)
    print(f"\n实验目录: {args.experiment_dir}")
    
    # 调试模式：显示目录结构
    if args.debug:
        print("\n" + "=" * 80)
        print("目录结构调试信息")
        print("=" * 80)
        
        prompt_dirs = sorted(Path(args.experiment_dir).glob("prompt_*"))
        print(f"\n找到 {len(prompt_dirs)} 个提示词目录\n")
        
        for prompt_dir in prompt_dirs[:3]:  # 只显示前3个
            print(f"\n{prompt_dir.name}:")
            
            # 显示variant目录
            for variant_dir in sorted(prompt_dir.glob("variant_*")):
                print(f"  {variant_dir.name}/")
                
                # 显示run目录
                for run_dir in sorted(variant_dir.glob("*")):
                    if run_dir.is_dir():
                        print(f"    {run_dir.name}/")
                        
                        # 显示stage目录
                        for stage_dir in sorted(run_dir.glob("stage*")):
                            if stage_dir.is_dir():
                                num_files = len(list(stage_dir.glob("*")))
                                print(f"      {stage_dir.name}/ ({num_files} files)")
        
        print("\n" + "=" * 80)
        return
    
    all_results = {}
    
    # 确定要计算哪些指标
    compute_fid = args.compute_fid or args.all
    compute_pc = args.compute_pointcloud or args.all
    compute_st = args.compute_stats or args.all
    
    # 计算FID
    if compute_fid:
        try:
            fid_results = compute_fid_between_variants(args.experiment_dir, args.device)
            all_results["fid"] = fid_results
        except Exception as e:
            print(f"\n错误: FID计算失败: {e}")
    
    # 计算点云质量
    if compute_pc:
        try:
            pc_results = compute_pointcloud_quality_all(args.experiment_dir)
            all_results["point_cloud_quality"] = pc_results
        except Exception as e:
            print(f"\n错误: 点云质量计算失败: {e}")
    
    # 计算统计显著性
    if compute_st:
        try:
            results_json = os.path.join(args.experiment_dir, "results.json")
            if os.path.exists(results_json):
                stats_results = compute_stats_from_results(results_json)
                all_results["statistical_significance"] = stats_results
            else:
                print(f"\n警告: results.json不存在: {results_json}")
        except Exception as e:
            print(f"\n错误: 统计显著性计算失败: {e}")
    
    # 保存结果（处理numpy类型）
    output_path = os.path.join(args.experiment_dir, "additional_metrics.json")
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ 结果已保存到: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
