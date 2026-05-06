#!/usr/bin/env python
"""
Progressive Comparison Experiment for Text2GS

Compares:
- Stage 1+2+4 (without ViewCrafter dense view synthesis)
- Stage 1+2+3+4 (full pipeline with ViewCrafter)

This experiment demonstrates the value of the multi-stage design,
specifically the contribution of Stage 3 (ViewCrafter) for dense view synthesis.
"""

import os
import sys
import argparse
import yaml
import json
import time
import random
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text2gs.pipeline import Text2GSPipeline
import subprocess
import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed / 1024  # Convert MB to GB
    except:
        pass
    
    # Fallback to torch
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
    except:
        pass
    
    return 0.0


def monitor_resources():
    """Monitor system resources"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_gb = psutil.virtual_memory().used / (1024**3)
        gpu_memory_gb = get_gpu_memory()
        
        return {
            "cpu_percent": cpu_percent,
            "ram_gb": memory_gb,
            "gpu_memory_gb": gpu_memory_gb
        }
    except ImportError:
        return {
            "cpu_percent": 0.0,
            "ram_gb": 0.0,
            "gpu_memory_gb": get_gpu_memory()
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive Comparison Experiment")
    
    parser.add_argument("--prompts-file", type=str, default="./experiments/prompts.txt",
                        help="Path to prompts file")
    parser.add_argument("--config", type=str, default="./configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--output", type=str, default="./experiments/results",
                        help="Output directory for experiments")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    # Experiment selection
    parser.add_argument("--run-124", action="store_true",
                        help="Run Stage 1+2+4 variant")
    parser.add_argument("--run-1234", action="store_true",
                        help="Run full pipeline (Stage 1+2+3+4)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all variants")
    
    # Prompt selection
    parser.add_argument("--complexity", type=str, default="all",
                        choices=["simple", "medium", "complex", "all"],
                        help="Which complexity level to test")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Number of prompts to test (default: all)")
    
    # Training options
    parser.add_argument("--train-3dgs", action="store_true",
                        help="Train 3D-GS after export")
    parser.add_argument("--gs-iterations", type=int, default=5000,
                        help="3D-GS training iterations")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation metrics after generation")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of times to run each experiment (for averaging)")
    
    # Resource monitoring
    parser.add_argument("--monitor-resources", action="store_true",
                        help="Monitor GPU/CPU resources during execution")
    
    return parser.parse_args()


def load_prompts(prompts_file: str, complexity: str = "all", num_prompts: int = None):
    """Load prompts from file"""
    prompts = []
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("|")
            if len(parts) != 2:
                continue
            
            level, prompt = parts[0].strip(), parts[1].strip()
            
            if complexity == "all" or level == complexity:
                prompts.append({"level": level, "prompt": prompt})
    
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    
    return prompts


def run_stage_124(prompt: str, output_dir: str, config: dict, args, run_id: int = 0):
    """Run Stage 1+2+4 variant (skip ViewCrafter)"""
    print("\n" + "=" * 80)
    print(f"Running Stage 1+2+4 (without ViewCrafter) - Run {run_id + 1}/{args.num_runs}")
    print("=" * 80)
    
    # Use run_stage124.py script
    cmd = [
        "python", "run_stage124.py",
        "--text", prompt,
        "--output", output_dir,
        "--config", args.config,
        "--device", args.device,
        "--unload-between-stages"
    ]
    
    if args.train_3dgs:
        cmd.append("--train-3dgs")
        cmd.extend(["--gs-iterations", str(args.gs_iterations)])
    
    # Monitor resources
    start_resources = monitor_resources() if args.monitor_resources else {}
    start_time = time.time()
    peak_gpu_memory = start_resources.get("gpu_memory_gb", 0.0)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    elapsed_time = time.time() - start_time
    end_resources = monitor_resources() if args.monitor_resources else {}
    
    # Track peak GPU memory (simplified - would need continuous monitoring for true peak)
    if args.monitor_resources:
        peak_gpu_memory = max(start_resources.get("gpu_memory_gb", 0), 
                              end_resources.get("gpu_memory_gb", 0))
    
    return {
        "success": result.returncode == 0,
        "elapsed_time": elapsed_time,
        "variant": "stage_124",
        "run_id": run_id,
        "resources": {
            "start": start_resources,
            "end": end_resources,
            "peak_gpu_memory_gb": peak_gpu_memory
        } if args.monitor_resources else {}
    }


def run_stage_1234(prompt: str, output_dir: str, config: dict, args, run_id: int = 0):
    """Run full pipeline (Stage 1+2+3+4)"""
    print("\n" + "=" * 80)
    print(f"Running Full Pipeline (Stage 1+2+3+4) - Run {run_id + 1}/{args.num_runs}")
    print("=" * 80)
    
    # Use main pipeline
    cmd = [
        "python", "-m", "text2gs.run",
        "--text", prompt,
        "--output", output_dir,
        "--config", args.config,
        "--device", args.device,
        "--unload-between-stages"
    ]
    
    if args.train_3dgs:
        cmd.append("--train-3dgs")
        cmd.extend(["--gs-iterations", str(args.gs_iterations)])
    
    # Monitor resources
    start_resources = monitor_resources() if args.monitor_resources else {}
    start_time = time.time()
    peak_gpu_memory = start_resources.get("gpu_memory_gb", 0.0)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    elapsed_time = time.time() - start_time
    end_resources = monitor_resources() if args.monitor_resources else {}
    
    # Track peak GPU memory
    if args.monitor_resources:
        peak_gpu_memory = max(start_resources.get("gpu_memory_gb", 0), 
                              end_resources.get("gpu_memory_gb", 0))
    
    return {
        "success": result.returncode == 0,
        "elapsed_time": elapsed_time,
        "variant": "stage_1234",
        "run_id": run_id,
        "resources": {
            "start": start_resources,
            "end": end_resources,
            "peak_gpu_memory_gb": peak_gpu_memory
        } if args.monitor_resources else {}
    }


def evaluate_results(output_dir: str, prompt: str, device: str):
    """Evaluate pipeline results"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation.metrics import evaluate_pipeline_results, save_evaluation_results
    
    print("\n" + "=" * 80)
    print("Evaluating Results")
    print("=" * 80)
    
    results = evaluate_pipeline_results(output_dir, prompt, device)
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    save_evaluation_results(results, eval_path)
    
    return results


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Load prompts
    prompts = load_prompts(args.prompts_file, args.complexity, args.num_prompts)
    print(f"\nLoaded {len(prompts)} prompts")
    
    # Determine which variants to run
    run_variants = []
    if args.run_all:
        run_variants = ["124", "1234"]
    else:
        if args.run_124:
            run_variants.append("124")
        if args.run_1234:
            run_variants.append("1234")
    
    if not run_variants:
        print("Error: No variants selected. Use --run-124, --run-1234, or --run-all")
        return
    
    print(f"Running variants: {run_variants}")
    print(f"Number of runs per experiment: {args.num_runs}")
    print(f"Random seed: {args.seed}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output, f"progressive_comparison_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration
    exp_config = {
        "timestamp": timestamp,
        "variants": run_variants,
        "prompts": prompts,
        "config_file": args.config,
        "train_3dgs": args.train_3dgs,
        "gs_iterations": args.gs_iterations,
        "evaluate": args.evaluate,
        "seed": args.seed,
        "num_runs": args.num_runs,
        "monitor_resources": args.monitor_resources
    }
    
    with open(os.path.join(exp_dir, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)
    
    # Run experiments
    all_results = []
    
    for prompt_idx, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        level = prompt_data["level"]
        
        print("\n" + "=" * 80)
        print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
        print(f"Complexity: {level}")
        print("=" * 80)
        
        prompt_results = {
            "prompt": prompt,
            "level": level,
            "variants": {}
        }
        
        for variant in run_variants:
            # Run multiple times for averaging
            variant_runs = []
            
            for run_id in range(args.num_runs):
                # Set different seed for each run
                run_seed = args.seed + run_id
                set_random_seed(run_seed)
                
                variant_dir = os.path.join(
                    exp_dir, 
                    f"prompt_{prompt_idx:02d}_{level}", 
                    f"variant_{variant}",
                    f"run_{run_id}"
                )
                os.makedirs(variant_dir, exist_ok=True)
                
                try:
                    if variant == "124":
                        result = run_stage_124(prompt, variant_dir, config, args, run_id)
                    elif variant == "1234":
                        result = run_stage_1234(prompt, variant_dir, config, args, run_id)
                    
                    # Evaluate if requested
                    if args.evaluate and result["success"]:
                        # Find the actual output directory (with timestamp)
                        subdirs = [d for d in os.listdir(variant_dir) if os.path.isdir(os.path.join(variant_dir, d))]
                        if subdirs:
                            actual_output = os.path.join(variant_dir, subdirs[0])
                            eval_results = evaluate_results(actual_output, prompt, args.device)
                            result["evaluation"] = eval_results
                except Exception as e:
                    print(f"\n❌ Error running {variant} for prompt {prompt_idx}: {e}")
                    result = {
                        "success": False,
                        "error": str(e),
                        "variant": variant,
                        "run_id": run_id,
                        "elapsed_time": 0.0
                    }
                
                variant_runs.append(result)
            
            # Aggregate results from multiple runs
            prompt_results["variants"][variant] = aggregate_runs(variant_runs)
            
            # Save after each variant completes (even if it failed)
            all_results_temp = all_results + [prompt_results]
            with open(os.path.join(exp_dir, "results.json"), "w") as f:
                json.dump(all_results_temp, f, indent=2)
        
        all_results.append(prompt_results)
        
        # Save intermediate results after both variants complete
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results, exp_dir)
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


def aggregate_runs(runs: list) -> dict:
    """Aggregate results from multiple runs"""
    if len(runs) == 1:
        return runs[0]
    
    # Aggregate timing
    times = [r["elapsed_time"] for r in runs if r["success"]]
    successes = sum(1 for r in runs if r["success"])
    
    aggregated = {
        "success": successes > 0,
        "num_runs": len(runs),
        "num_successes": successes,
        "elapsed_time_mean": float(np.mean(times)) if times else 0.0,
        "elapsed_time_std": float(np.std(times)) if times else 0.0,
        "elapsed_time_min": float(np.min(times)) if times else 0.0,
        "elapsed_time_max": float(np.max(times)) if times else 0.0,
        "variant": runs[0]["variant"],
        "runs": runs  # Keep individual run data
    }
    
    # Aggregate evaluation metrics if available
    if "evaluation" in runs[0]:
        eval_metrics = {}
        for metric_name in ["multi_view_consistency", "text_image_alignment", "rendering_quality"]:
            if metric_name in runs[0]["evaluation"]:
                # Get the main score for this metric
                if metric_name == "multi_view_consistency":
                    score_key = "mean_similarity"
                elif metric_name == "text_image_alignment":
                    score_key = "mean_clip_score"
                else:
                    score_key = "mean_quality"
                
                scores = [
                    r["evaluation"][metric_name][score_key] 
                    for r in runs 
                    if "evaluation" in r and metric_name in r["evaluation"]
                ]
                
                if scores:
                    eval_metrics[metric_name] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores))
                    }
        
        if eval_metrics:
            aggregated["evaluation_aggregated"] = eval_metrics
    
    return aggregated


def generate_summary_report(results, exp_dir):
    """Generate summary report"""
    report_path = os.path.join(exp_dir, "SUMMARY_REPORT.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PROGRESSIVE COMPARISON EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Prompts: {len(results)}\n")
        f.write(f"Experiment Directory: {exp_dir}\n\n")
        
        # Count by complexity
        complexity_counts = {}
        for r in results:
            level = r["level"]
            complexity_counts[level] = complexity_counts.get(level, 0) + 1
        
        f.write("Prompts by Complexity:\n")
        for level, count in sorted(complexity_counts.items()):
            f.write(f"  {level}: {count}\n")
        f.write("\n")
        
        # Variant statistics
        f.write("-" * 80 + "\n")
        f.write("VARIANT STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        for variant in ["124", "1234"]:
            f.write(f"Variant: Stage {variant}\n")
            
            times = []
            successes = 0
            
            for r in results:
                if variant in r["variants"]:
                    v = r["variants"][variant]
                    if v["success"]:
                        successes += 1
                        times.append(v["elapsed_time"])
            
            f.write(f"  Success Rate: {successes}/{len(results)}\n")
            if times:
                f.write(f"  Average Time: {sum(times)/len(times):.2f} seconds\n")
                f.write(f"  Min Time: {min(times):.2f} seconds\n")
                f.write(f"  Max Time: {max(times):.2f} seconds\n")
            f.write("\n")
        
        # Evaluation metrics (if available)
        f.write("-" * 80 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        for variant in ["124", "1234"]:
            f.write(f"Variant: Stage {variant}\n")
            
            mvc_scores = []
            tia_scores = []
            
            for r in results:
                if variant in r["variants"]:
                    v = r["variants"][variant]
                    if "evaluation" in v:
                        eval_data = v["evaluation"]
                        if "multi_view_consistency" in eval_data:
                            mvc_scores.append(eval_data["multi_view_consistency"]["mean_similarity"])
                        if "text_image_alignment" in eval_data:
                            tia_scores.append(eval_data["text_image_alignment"]["mean_clip_score"])
            
            if mvc_scores:
                f.write(f"  Multi-view Consistency: {sum(mvc_scores)/len(mvc_scores):.4f}\n")
            if tia_scores:
                f.write(f"  Text-Image Alignment: {sum(tia_scores)/len(tia_scores):.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("For detailed results, see results.json\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSummary report saved to: {report_path}")


if __name__ == "__main__":
    main()
