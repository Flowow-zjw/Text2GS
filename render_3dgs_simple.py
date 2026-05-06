#!/usr/bin/env python3
"""
简化版3D-GS渲染脚本
生成渲染命令和相机轨迹文件，可以手动或批量执行
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rendering commands for 3D-GS models")
    
    parser.add_argument("--results-json", type=str, default="./results.json",
                        help="Path to results.json from experiments")
    parser.add_argument("--output-dir", type=str, default="./rendered_results",
                        help="Output directory for rendered images")
    parser.add_argument("--gs-path", type=str, default="/root/autodl-tmp/gaussian-splatting",
                        help="Path to gaussian-splatting installation")
    parser.add_argument("--generate-script", action="store_true",
                        help="Generate a bash script to run all renderings")
    
    return parser.parse_args()


def load_results(results_path: str):
    """加载实验结果"""
    with open(results_path, 'r') as f:
        return json.load(f)


def find_model_path(variant_data: dict) -> str:
    """从variant数据中找到训练好的模型路径"""
    if "evaluation" in variant_data:
        eval_data = variant_data["evaluation"]
        if "stage4_metadata" in eval_data:
            stage4 = eval_data["stage4_metadata"]
            if "training" in stage4:
                model_path = stage4["training"].get("model_path")
                if model_path and os.path.exists(model_path):
                    return model_path
    return None


def find_source_path(variant_data: dict) -> str:
    """找到source path (包含images和sparse目录的父目录)"""
    if "evaluation" in variant_data:
        eval_data = variant_data["evaluation"]
        if "stage4_metadata" in eval_data:
            stage4 = eval_data["stage4_metadata"]
            export_dir = stage4.get("export_dir")
            if export_dir:
                # 转换为绝对路径
                abs_path = os.path.abspath(export_dir)
                if os.path.exists(abs_path):
                    return abs_path
    return None


def main():
    args = parse_args()
    
    print("=" * 80)
    print("3D-GS Rendering Command Generator")
    print("=" * 80)
    
    # 加载实验结果
    results = load_results(args.results_json)
    print(f"\nFound {len(results)} experiments")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有渲染任务
    render_tasks = []
    
    for prompt_idx, prompt_data in enumerate(results):
        prompt = prompt_data["prompt"]
        level = prompt_data.get("level", "unknown")
        
        for variant in ["124", "1234"]:
            if variant not in prompt_data.get("variants", {}):
                continue
            
            variant_data = prompt_data["variants"][variant]
            
            # 查找模型路径
            model_path = find_model_path(variant_data)
            if not model_path:
                print(f"⚠️  Prompt {prompt_idx}, Variant {variant}: Model not found")
                continue
            
            # 查找source path
            source_path = find_source_path(variant_data)
            if not source_path:
                print(f"⚠️  Prompt {prompt_idx}, Variant {variant}: Source path not found")
                continue
            
            # 输出目录
            output_dir = os.path.join(args.output_dir, f"prompt_{prompt_idx:02d}_{level}", f"variant_{variant}")
            
            render_tasks.append({
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "level": level,
                "variant": variant,
                "model_path": model_path,
                "source_path": source_path,
                "output_dir": output_dir
            })
    
    print(f"\nFound {len(render_tasks)} models to render")
    
    # 保存任务列表
    tasks_file = os.path.join(args.output_dir, "render_tasks.json")
    with open(tasks_file, 'w') as f:
        json.dump(render_tasks, f, indent=2)
    print(f"\nTasks saved to: {tasks_file}")
    
    # 生成渲染命令
    commands = []
    
    print("\n" + "=" * 80)
    print("RENDERING COMMANDS")
    print("=" * 80)
    
    for task in render_tasks:
        # 使用gaussian-splatting的render.py
        # 命令格式: python render.py -m <model_path> -s <source_path>
        
        # 渲染训练集视角（大部分图像在训练集）
        # 使用 --skip_test 只渲染训练集
        cmd = (
            f"python {os.path.join(args.gs_path, 'render.py')} "
            f"-m {task['model_path']} "
            f"-s {task['source_path']} "
            f"--skip_test"
        )
        
        commands.append({
            "task": task,
            "command": cmd
        })
        
        print(f"\n[Prompt {task['prompt_idx']:02d} - Variant {task['variant']}]")
        print(f"  Prompt: {task['prompt'][:60]}...")
        print(f"  Model: {task['model_path']}")
        print(f"  Command: {cmd}")
    
    # 生成bash脚本
    if args.generate_script:
        script_path = os.path.join(args.output_dir, "render_all.sh")
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated rendering script\n")
            f.write(f"# Generated from: {args.results_json}\n")
            f.write("# IMPORTANT: Run this in the 'text2gs' conda environment!\n")
            f.write("# Usage: conda activate text2gs && bash render_all.sh\n\n")
            f.write("set -e\n\n")
            f.write("# Check if we're in the correct conda environment\n")
            f.write('if [[ "$CONDA_DEFAULT_ENV" != "text2gs" ]]; then\n')
            f.write('    echo "ERROR: Please activate the \'text2gs\' conda environment first!"\n')
            f.write('    echo "Run: conda activate text2gs"\n')
            f.write('    exit 1\n')
            f.write('fi\n\n')
            
            for i, cmd_data in enumerate(commands):
                task = cmd_data["task"]
                cmd = cmd_data["command"]
                
                f.write(f"# Task {i+1}/{len(commands)}: Prompt {task['prompt_idx']:02d} - Variant {task['variant']}\n")
                f.write(f"echo 'Rendering [{i+1}/{len(commands)}] Prompt {task['prompt_idx']:02d} - Variant {task['variant']}'\n")
                f.write(f"{cmd}\n\n")
            
            f.write("echo 'All renderings complete!'\n")
        
        # 使脚本可执行
        os.chmod(script_path, 0o755)
        
        print(f"\n{'='*80}")
        print(f"Bash script generated: {script_path}")
        print(f"Run with: bash {script_path}")
        print(f"{'='*80}")
    
    # 保存命令列表
    commands_file = os.path.join(args.output_dir, "render_commands.json")
    with open(commands_file, 'w') as f:
        json.dump(commands, f, indent=2)
    print(f"\nCommands saved to: {commands_file}")
    
    # 生成Python批量执行脚本
    python_script = os.path.join(args.output_dir, "run_rendering.py")
    with open(python_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
批量执行渲染任务
"""

import os
import sys
import json
import subprocess
from tqdm import tqdm

# 加载任务
with open("render_commands.json", "r") as f:
    commands = json.load(f)

print(f"Total tasks: {len(commands)}")

failed_tasks = []

for i, cmd_data in enumerate(tqdm(commands, desc="Rendering")):
    task = cmd_data["task"]
    cmd = cmd_data["command"]
    
    print(f"\\n[{i+1}/{len(commands)}] Prompt {task['prompt_idx']:02d} - Variant {task['variant']}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ Success")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        failed_tasks.append({
            "task": task,
            "error": str(e)
        })

print(f"\\n{'='*80}")
print(f"Rendering complete!")
print(f"  Successful: {len(commands) - len(failed_tasks)}/{len(commands)}")
print(f"  Failed: {len(failed_tasks)}/{len(commands)}")

if failed_tasks:
    print(f"\\nFailed tasks:")
    for ft in failed_tasks:
        print(f"  - Prompt {ft['task']['prompt_idx']:02d} Variant {ft['task']['variant']}")
    
    with open("failed_tasks.json", "w") as f:
        json.dump(failed_tasks, f, indent=2)
    print(f"\\nFailed tasks saved to: failed_tasks.json")

print(f"{'='*80}")
''')
    
    os.chmod(python_script, 0o755)
    print(f"Python script generated: {python_script}")
    print(f"Run with: cd {args.output_dir} && python run_rendering.py")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the generated commands")
    print(f"2. Run the bash script: bash {script_path}")
    print(f"   OR run the Python script: cd {args.output_dir} && python run_rendering.py")
    print("\n3. After rendering, the images will be in:")
    print(f"   <model_path>/train/ours_<iteration>/renders/")
    print(f"   (e.g., .../3dgs/output/train/ours_7000/renders/)")
    print("\n4. Use these rendered images for evaluation")
    print("=" * 80)


if __name__ == "__main__":
    main()
