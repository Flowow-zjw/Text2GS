#!/usr/bin/env python
"""
Analyze and visualize experimental results
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("results_dir", type=str,
                        help="Path to experiment results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots (default: same as results_dir)")
    return parser.parse_args()


def load_results(results_dir):
    """Load results.json from experiment directory"""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r") as f:
        return json.load(f)


def extract_metrics(results):
    """Extract metrics from results for comparison"""
    data = {
        "124": {
            "times": [],
            "mvc": [],  # multi-view consistency
            "tia": [],  # text-image alignment
            "quality": [],
            "prompts": [],
            "levels": []  # complexity levels
        },
        "1234": {
            "times": [],
            "mvc": [],
            "tia": [],
            "quality": [],
            "prompts": [],
            "levels": []
        }
    }
    
    for prompt_result in results:
        prompt = prompt_result["prompt"]
        level = prompt_result.get("level", "unknown")
        
        for variant in ["124", "1234"]:
            if variant not in prompt_result["variants"]:
                continue
            
            v = prompt_result["variants"][variant]
            
            if not v.get("success", False):
                continue
            
            data[variant]["prompts"].append(prompt)
            data[variant]["levels"].append(level)
            
            # Handle aggregated results (multiple runs)
            if "elapsed_time_mean" in v:
                data[variant]["times"].append(v["elapsed_time_mean"])
            else:
                data[variant]["times"].append(v.get("elapsed_time", 0))
            
            # Extract evaluation metrics
            if "evaluation_aggregated" in v:
                # Aggregated metrics from multiple runs
                eval_data = v["evaluation_aggregated"]
                
                if "multi_view_consistency" in eval_data:
                    data[variant]["mvc"].append(eval_data["multi_view_consistency"]["mean"])
                else:
                    data[variant]["mvc"].append(None)
                
                if "text_image_alignment" in eval_data:
                    data[variant]["tia"].append(eval_data["text_image_alignment"]["mean"])
                else:
                    data[variant]["tia"].append(None)
                
                if "rendering_quality" in eval_data:
                    data[variant]["quality"].append(eval_data["rendering_quality"]["mean"])
                else:
                    data[variant]["quality"].append(None)
                    
            elif "evaluation" in v:
                # Single run evaluation
                eval_data = v["evaluation"]
                
                if "multi_view_consistency" in eval_data:
                    mvc = eval_data["multi_view_consistency"]["mean_similarity"]
                    data[variant]["mvc"].append(mvc)
                else:
                    data[variant]["mvc"].append(None)
                
                if "text_image_alignment" in eval_data:
                    tia = eval_data["text_image_alignment"]["mean_clip_score"]
                    data[variant]["tia"].append(tia)
                else:
                    data[variant]["tia"].append(None)
                
                if "rendering_quality" in eval_data:
                    quality = eval_data["rendering_quality"]["mean_quality"]
                    data[variant]["quality"].append(quality)
                else:
                    data[variant]["quality"].append(None)
            else:
                data[variant]["mvc"].append(None)
                data[variant]["tia"].append(None)
                data[variant]["quality"].append(None)
    
    return data


def plot_comparison(data, output_dir):
    """Generate comparison plots"""
    
    # Filter out None values
    def filter_none(lst):
        return [x for x in lst if x is not None]
    
    # 1. Execution Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times_124 = data["124"]["times"]
    times_1234 = data["1234"]["times"]
    
    x = np.arange(len(times_124))
    width = 0.35
    
    ax.bar(x - width/2, times_124, width, label='Stage 1+2+4', alpha=0.8)
    ax.bar(x + width/2, times_1234, width, label='Stage 1+2+3+4', alpha=0.8)
    
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_time_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Multi-view Consistency Comparison
    mvc_124 = filter_none(data["124"]["mvc"])
    mvc_1234 = filter_none(data["1234"]["mvc"])
    
    if mvc_124 and mvc_1234:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(mvc_124))
        width = 0.35
        
        ax.bar(x - width/2, mvc_124, width, label='Stage 1+2+4', alpha=0.8)
        ax.bar(x + width/2, mvc_1234, width, label='Stage 1+2+3+4', alpha=0.8)
        
        ax.set_xlabel('Prompt Index')
        ax.set_ylabel('Multi-view Consistency Score')
        ax.set_title('Multi-view Consistency Comparison')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multi_view_consistency_comparison.png"), dpi=300)
        plt.close()
    
    # 3. Text-Image Alignment Comparison
    tia_124 = filter_none(data["124"]["tia"])
    tia_1234 = filter_none(data["1234"]["tia"])
    
    if tia_124 and tia_1234:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(tia_124))
        width = 0.35
        
        ax.bar(x - width/2, tia_124, width, label='Stage 1+2+4', alpha=0.8)
        ax.bar(x + width/2, tia_1234, width, label='Stage 1+2+3+4', alpha=0.8)
        
        ax.set_xlabel('Prompt Index')
        ax.set_ylabel('CLIP Score')
        ax.set_title('Text-Image Alignment Comparison')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "text_image_alignment_comparison.png"), dpi=300)
        plt.close()
    
    # 4. Summary Statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time statistics
    ax = axes[0, 0]
    ax.boxplot([times_124, times_1234], labels=['Stage 1+2+4', 'Stage 1+2+3+4'])
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # MVC statistics
    if mvc_124 and mvc_1234:
        ax = axes[0, 1]
        ax.boxplot([mvc_124, mvc_1234], labels=['Stage 1+2+4', 'Stage 1+2+3+4'])
        ax.set_ylabel('Multi-view Consistency')
        ax.set_title('Multi-view Consistency Distribution')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    # TIA statistics
    if tia_124 and tia_1234:
        ax = axes[1, 0]
        ax.boxplot([tia_124, tia_1234], labels=['Stage 1+2+4', 'Stage 1+2+3+4'])
        ax.set_ylabel('CLIP Score')
        ax.set_title('Text-Image Alignment Distribution')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    # Average comparison
    ax = axes[1, 1]
    metrics = ['Time\n(lower better)', 'MVC\n(higher better)', 'TIA\n(higher better)']
    
    # Normalize time (inverse for "lower is better")
    time_124_norm = 1.0 / (np.mean(times_124) / 60.0) if times_124 else 0
    time_1234_norm = 1.0 / (np.mean(times_1234) / 60.0) if times_1234 else 0
    
    values_124 = [
        time_124_norm,
        np.mean(mvc_124) if mvc_124 else 0,
        np.mean(tia_124) if tia_124 else 0
    ]
    values_1234 = [
        time_1234_norm,
        np.mean(mvc_1234) if mvc_1234 else 0,
        np.mean(tia_1234) if tia_1234 else 0
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, values_124, width, label='Stage 1+2+4', alpha=0.8)
    ax.bar(x + width/2, values_1234, width, label='Stage 1+2+3+4', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized Score')
    ax.set_title('Average Performance Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_statistics.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def print_statistics(data):
    """Print statistical summary"""
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80 + "\n")
    
    for variant in ["124", "1234"]:
        print(f"Variant: Stage {variant}")
        print("-" * 40)
        
        times = data[variant]["times"]
        mvc = [x for x in data[variant]["mvc"] if x is not None]
        tia = [x for x in data[variant]["tia"] if x is not None]
        
        if times:
            print(f"  Execution Time:")
            print(f"    Mean: {np.mean(times):.2f}s")
            print(f"    Std:  {np.std(times):.2f}s")
            print(f"    Min:  {np.min(times):.2f}s")
            print(f"    Max:  {np.max(times):.2f}s")
        
        if mvc:
            print(f"  Multi-view Consistency:")
            print(f"    Mean: {np.mean(mvc):.4f}")
            print(f"    Std:  {np.std(mvc):.4f}")
            print(f"    Min:  {np.min(mvc):.4f}")
            print(f"    Max:  {np.max(mvc):.4f}")
        
        if tia:
            print(f"  Text-Image Alignment:")
            print(f"    Mean: {np.mean(tia):.4f}")
            print(f"    Std:  {np.std(tia):.4f}")
            print(f"    Min:  {np.min(tia):.4f}")
            print(f"    Max:  {np.max(tia):.4f}")
        
        print()
    
    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80 + "\n")
    
    times_124 = data["124"]["times"]
    times_1234 = data["1234"]["times"]
    
    if times_124 and times_1234:
        speedup = np.mean(times_1234) / np.mean(times_124)
        print(f"Stage 1+2+3+4 is {speedup:.2f}x slower than Stage 1+2+4")
        print(f"  (Stage 1+2+4: {np.mean(times_124):.2f}s, Stage 1+2+3+4: {np.mean(times_1234):.2f}s)")
    
    mvc_124 = [x for x in data["124"]["mvc"] if x is not None]
    mvc_1234 = [x for x in data["1234"]["mvc"] if x is not None]
    
    if mvc_124 and mvc_1234:
        improvement = (np.mean(mvc_1234) - np.mean(mvc_124)) / np.mean(mvc_124) * 100
        print(f"\nMulti-view Consistency improvement: {improvement:+.2f}%")
        print(f"  (Stage 1+2+4: {np.mean(mvc_124):.4f}, Stage 1+2+3+4: {np.mean(mvc_1234):.4f})")
    
    tia_124 = [x for x in data["124"]["tia"] if x is not None]
    tia_1234 = [x for x in data["1234"]["tia"] if x is not None]
    
    if tia_124 and tia_1234:
        improvement = (np.mean(tia_1234) - np.mean(tia_124)) / np.mean(tia_124) * 100
        print(f"\nText-Image Alignment improvement: {improvement:+.2f}%")
        print(f"  (Stage 1+2+4: {np.mean(tia_124):.4f}, Stage 1+2+3+4: {np.mean(tia_1234):.4f})")
    
    print()


def main():
    args = parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    # Extract metrics
    print("Extracting metrics...")
    data = extract_metrics(results)
    
    # Print statistics
    print_statistics(data)
    
    # Generate plots
    output_dir = args.output if args.output else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots...")
    plot_comparison(data, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
