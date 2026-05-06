#!/usr/bin/env python3
"""
评估渲染后的3D-GS图像
使用与之前相同的评估指标，但基于渲染的图像而不是Stage 1的图像
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict

# 添加evaluation模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.metrics import (
    compute_multi_view_consistency,
    compute_text_image_alignment,
    compute_rendering_quality,
    compute_lpips_consistency,
    compute_statistical_significance
)


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rendered 3D-GS images")
    
    parser.add_argument("--rendered-dir", type=str, default="./rendered_results",
                        help="Directory containing rendered images")
    parser.add_argument("--results-json", type=str, default="./results.json",
                        help="Original results.json for prompts")
    parser.add_argument("--output", type=str, default="./rendered_evaluation.json",
                        help="Output file for evaluation results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for evaluation")
    parser.add_argument("--iteration", type=int, default=7000,
                        help="Which iteration to evaluate (default: 7000)")
    
    return parser.parse_args()


def load_rendered_images(model_path: str, iteration: int = 7000) -> List[np.ndarray]:
    """
    加载渲染的图像
    gaussian-splatting的输出格式: <model_path>/train/ours_<iteration>/renders/00000.png, 00001.png, ...
    """
    images_dir = os.path.join(model_path, "train", f"ours_{iteration}", "renders")
    
    if not os.path.exists(images_dir):
        print(f"  Warning: Rendered images not found in {images_dir}")
        return []
    
    # 加载所有PNG图像
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    images = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = np.array(Image.open(img_path))
        images.append(img)
    
    return images


def evaluate_variant(
    model_path: str,
    prompt: str,
    device: str,
    iteration: int = 7000
) -> Dict:
    """评估单个variant的渲染结果"""
    
    # 加载渲染图像
    images = load_rendered_images(model_path, iteration)
    
    if not images:
        return {
            "success": False,
            "error": "no_rendered_images",
            "model_path": model_path
        }
    
    print(f"    Loaded {len(images)} rendered images")
    
    # 计算评估指标
    results = {
        "success": True,
        "model_path": model_path,
        "num_images": len(images),
        "iteration": iteration
    }
    
    try:
        # 1. Multi-view Consistency
        print(f"    Computing multi-view consistency...")
        mvc = compute_multi_view_consistency(images, device)
        results["multi_view_consistency"] = mvc
        
        # 2. Text-Image Alignment
        print(f"    Computing text-image alignment...")
        tia = compute_text_image_alignment(prompt, images, device)
        results["text_image_alignment"] = tia
        
        # 3. Rendering Quality
        print(f"    Computing rendering quality...")
        rq = compute_rendering_quality(images)
        results["rendering_quality"] = rq
        
        # 4. LPIPS Consistency
        print(f"    Computing LPIPS consistency...")
        lpips = compute_lpips_consistency(images, device)
        results["lpips_consistency"] = lpips
        
    except Exception as e:
        print(f"    Error during evaluation: {e}")
        results["success"] = False
        results["error"] = str(e)
    
    return results


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Rendered Images Evaluation")
    print("=" * 80)
    print(f"\nRendered directory: {args.rendered_dir}")
    print(f"Results file: {args.results_json}")
    print(f"Iteration: {args.iteration}")
    print(f"Device: {args.device}")
    
    # 加载原始结果（获取prompts）
    with open(args.results_json, 'r') as f:
        original_results = json.load(f)
    
    print(f"\nFound {len(original_results)} experiments")
    
    # 评估每个实验
    evaluation_results = []
    
    for prompt_idx, prompt_data in enumerate(original_results):
        prompt = prompt_data["prompt"]
        level = prompt_data.get("level", "unknown")
        
        print(f"\n{'='*80}")
        print(f"[{prompt_idx + 1}/{len(original_results)}] {prompt}")
        print(f"Level: {level}")
        print(f"{'='*80}")
        
        prompt_result = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "level": level,
            "variants": {}
        }
        
        for variant in ["124", "1234"]:
            print(f"\n  Variant {variant}:")
            
            # 从原始结果中获取模型路径
            if variant not in prompt_data.get("variants", {}):
                print(f"    ✗ Variant {variant} not found in results")
                prompt_result["variants"][variant] = {
                    "success": False,
                    "error": "variant_not_found"
                }
                continue
            
            variant_data = prompt_data["variants"][variant]
            
            # 提取模型路径
            model_path = None
            if "evaluation" in variant_data:
                eval_data = variant_data["evaluation"]
                if "stage4_metadata" in eval_data:
                    stage4 = eval_data["stage4_metadata"]
                    if "training" in stage4:
                        model_path = stage4["training"].get("model_path")
            
            if not model_path or not os.path.exists(model_path):
                print(f"    ✗ Model path not found: {model_path}")
                prompt_result["variants"][variant] = {
                    "success": False,
                    "error": "model_path_not_found"
                }
                continue
            
            # 评估
            eval_result = evaluate_variant(model_path, prompt, args.device, args.iteration)
            prompt_result["variants"][variant] = eval_result
            
            if eval_result["success"]:
                print(f"    ✓ Evaluation complete")
                print(f"      MVC: {eval_result['multi_view_consistency']['mean_similarity']:.4f}")
                print(f"      TIA: {eval_result['text_image_alignment']['mean_clip_score']:.4f}")
                print(f"      Quality: {eval_result['rendering_quality']['mean_quality']:.4f}")
                print(f"      LPIPS: {eval_result['lpips_consistency']['mean_lpips_consistency']:.4f}")
            else:
                print(f"    ✗ Evaluation failed: {eval_result.get('error', 'unknown')}")
        
        evaluation_results.append(prompt_result)
        
        # 保存中间结果（转换 numpy 类型）
        with open(args.output, 'w') as f:
            json.dump(convert_to_json_serializable(evaluation_results), f, indent=2)
    
    # 计算统计显著性
    print("\n" + "=" * 80)
    print("COMPUTING STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    
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
    
    for result in evaluation_results:
        for variant in ['124', '1234']:
            if variant in result['variants'] and result['variants'][variant].get('success'):
                eval_data = result['variants'][variant]
                
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
    
    # 计算统计检验
    statistical_results = {}
    
    for metric_key, metric_name in [
        ('clip_mvc', 'CLIP Multi-view Consistency'),
        ('clip_text', 'CLIP Text-Image Alignment'),
        ('rendering_quality', 'Rendering Quality'),
        ('lpips', 'LPIPS Consistency')
    ]:
        data_124 = metrics_data['124'][metric_key]
        data_1234 = metrics_data['1234'][metric_key]
        
        if len(data_124) > 0 and len(data_1234) > 0:
            print(f"\n{metric_name}:")
            print(f"  124:  n={len(data_124)}, mean={np.mean(data_124):.4f}")
            print(f"  1234: n={len(data_1234)}, mean={np.mean(data_1234):.4f}")
            
            stats = compute_statistical_significance(data_124, data_1234)
            
            # Convert numpy types to Python types for JSON serialization
            stats_json = {
                'p_value': float(stats['p_value']),
                'cohen_d': float(stats['cohen_d']),
                'effect_size': str(stats['effect_size']),
                'significant_at_0.05': bool(stats['significant_at_0.05'])
            }
            statistical_results[metric_key] = stats_json
            
            print(f"  p-value: {stats['p_value']:.4f}")
            print(f"  Cohen's d: {stats['cohen_d']:.4f} ({stats['effect_size']})")
            print(f"  Significant: {stats['significant_at_0.05']}")
    
    # 保存最终结果
    final_results = {
        "evaluation_results": evaluation_results,
        "statistical_significance": statistical_results,
        "summary": {
            "num_experiments": len(evaluation_results),
            "num_successful_124": sum(1 for r in evaluation_results 
                                      if r['variants'].get('124', {}).get('success')),
            "num_successful_1234": sum(1 for r in evaluation_results 
                                       if r['variants'].get('1234', {}).get('success')),
            "iteration": args.iteration
        }
    }
    
    # Convert all numpy types to Python native types
    final_results = convert_to_json_serializable(final_results)
    
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal experiments: {len(evaluation_results)}")
    print(f"  Successful 124: {final_results['summary']['num_successful_124']}")
    print(f"  Successful 1234: {final_results['summary']['num_successful_1234']}")
    
    print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
