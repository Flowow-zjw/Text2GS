#!/usr/bin/env python
"""
从 Stage 3 的输出直接开始训练 3D-GS
跳过 Stage 1-3，节省时间

从 stage3_viewcrafter/ 读取数据，生成 COLMAP 格式，然后训练

用法:
    python train_from_stage3.py --run-dir output/20260412_120000
"""

import os
import sys
import argparse
import yaml
import subprocess
import numpy as np
from datetime import datetime
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 Stage 3 输出直接训练 3D-GS"
    )
    
    # Required
    parser.add_argument(
        "--run-dir", 
        type=str, 
        required=True,
        help="完整运行目录路径（例如：output/20260412_120000，包含 stage3_viewcrafter/ 子目录）"
    )
    
    # Optional
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="训练迭代次数（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="球谐函数阶数 (0-3)，控制色彩表现"
    )
    
    parser.add_argument(
        "--lambda-dssim",
        type=float,
        default=None,
        help="SSIM 损失权重 (0-1)"
    )
    
    parser.add_argument(
        "--opacity-reset-interval",
        type=int,
        default=None,
        help="透明度重置间隔（设置大值如100000可禁用）"
    )
    
    parser.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=None,
        help="密集化梯度阈值"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="训练完成后压缩结果"
    )
    
    parser.add_argument(
        "--compress-mode",
        type=str,
        default="model",
        choices=["model", "full"],
        help="压缩模式: model (仅模型) 或 full (完整目录)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def export_colmap_from_stage3(stage3_dir: str, output_dir: str, max_points: int = 200000) -> dict:
    """
    从 Stage 3 输出生成 COLMAP 格式数据
    
    Args:
        stage3_dir: stage3_viewcrafter 目录
        output_dir: 输出目录（3dgs/）
        max_points: 最大点云数量
        
    Returns:
        导出信息字典
    """
    print(f"\n{'='*80}")
    print("从 Stage 3 数据生成 COLMAP 格式")
    print(f"{'='*80}")
    print(f"Stage 3 目录: {stage3_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # 创建目录
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # 1. 加载相机参数
    print("  [1/4] 加载相机参数...")
    cameras_file = os.path.join(stage3_dir, "cameras.npz")
    if not os.path.exists(cameras_file):
        raise FileNotFoundError(f"相机参数文件不存在: {cameras_file}")
    
    cameras = np.load(cameras_file)
    c2ws = cameras["c2ws"]  # (N, 4, 4)
    focals = cameras["focals"]  # (N, 2)
    principal_points = cameras["principal_points"]  # (N, 2)
    
    num_views = len(c2ws)
    print(f"    ✓ 加载 {num_views} 个相机")
    
    # 2. 复制重建图像
    print("  [2/4] 复制重建图像...")
    reconstructed_images_dir = os.path.join(stage3_dir, "reconstructed_images")
    if not os.path.exists(reconstructed_images_dir):
        raise FileNotFoundError(f"重建图像目录不存在: {reconstructed_images_dir}")
    
    image_files = sorted([f for f in os.listdir(reconstructed_images_dir) if f.endswith('.png')])
    image_names = []
    
    # 获取图像尺寸
    first_img = Image.open(os.path.join(reconstructed_images_dir, image_files[0]))
    W, H = first_img.size
    print(f"    图像分辨率: {W}x{H}")
    
    for i, img_file in enumerate(image_files):
        src = os.path.join(reconstructed_images_dir, img_file)
        dst_name = f"image_{i:04d}.png"
        dst = os.path.join(images_dir, dst_name)
        
        # 复制图像
        img = Image.open(src)
        img.save(dst)
        image_names.append(dst_name)
    
    print(f"    ✓ 复制 {len(image_names)} 张图像")
    
    # 3. 加载点云
    print("  [3/4] 加载点云...")
    pointcloud_file = os.path.join(stage3_dir, "pointcloud_reconstructed.ply")
    if not os.path.exists(pointcloud_file):
        raise FileNotFoundError(f"点云文件不存在: {pointcloud_file}")
    
    import trimesh
    pcd = trimesh.load(pointcloud_file)
    all_pts = pcd.vertices
    all_cols = pcd.colors[:, :3]  # RGB
    
    # 采样点云
    if len(all_pts) > max_points:
        step = len(all_pts) // max_points
        all_pts = all_pts[::step]
        all_cols = all_cols[::step]
    
    print(f"    ✓ 加载 {len(all_pts)} 个点")
    
    # 4. 写入 COLMAP 文件
    print("  [4/4] 写入 COLMAP 文件...")
    
    # cameras.txt
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # 使用第一个相机的参数（假设所有相机参数相同）
        focal = focals[0]
        pp = principal_points[0]
        
        # 检查 focal 的形状
        if isinstance(focal, np.ndarray):
            if focal.shape == (2,):
                fx, fy = focal[0], focal[1]
            elif focal.shape == ():
                fx = fy = float(focal)
            else:
                fx = fy = float(focal.flatten()[0])
        else:
            fx = fy = float(focal)
        
        # 检查 principal_points 的形状
        if isinstance(pp, np.ndarray):
            if pp.shape == (2,):
                cx, cy = pp[0], pp[1]
            elif pp.shape == ():
                cx = cy = float(pp)
            else:
                cx = cy = float(pp.flatten()[0])
        else:
            cx = cy = float(pp)
        
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    
    print(f"    ✓ cameras.txt (fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f})")
    
    # images.txt
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (name, c2w) in enumerate(zip(image_names, c2ws)):
            # c2w -> w2c
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # 旋转矩阵 -> 四元数
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n")
            f.write("\n")
    
    print(f"    ✓ images.txt")
    
    # points3D.txt
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        
        for i, (pt, col) in enumerate(zip(all_pts, all_cols)):
            r, g, b = int(col[0]), int(col[1]), int(col[2])
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0\n")
    
    print(f"    ✓ points3D.txt")
    
    print(f"\n{'='*80}")
    print("COLMAP 格式生成完成")
    print(f"{'='*80}\n")
    
    return {
        "num_images": len(image_names),
        "num_points": len(all_pts),
        "resolution": [H, W]
    }


def train_3dgs(data_dir: str, config: dict, args) -> dict:
    """
    调用 gaussian-splatting 训练
    
    Args:
        data_dir: COLMAP 数据目录（3dgs_retrain_xxx/）
        config: 配置字典
        args: 命令行参数
        
    Returns:
        训练结果字典
    """
    # 转换为绝对路径
    data_dir = os.path.abspath(data_dir)
    
    # 获取配置
    gaussian_config = config.get("gaussian", {})
    paths_config = config.get("paths", {})
    
    gs_path = paths_config.get("gaussian_splatting_path", "/root/autodl-tmp/gaussian-splatting")
    
    # 训练参数（优先级：命令行 > 配置文件 > 默认值）
    iterations = args.iterations or gaussian_config.get("iterations", 5000)
    sh_degree = args.sh_degree if args.sh_degree is not None else gaussian_config.get("sh_degree")
    lambda_dssim = args.lambda_dssim if args.lambda_dssim is not None else gaussian_config.get("lambda_dssim")
    opacity_reset_interval = args.opacity_reset_interval if args.opacity_reset_interval is not None else gaussian_config.get("opacity_reset_interval")
    densify_grad_threshold = args.densify_grad_threshold if args.densify_grad_threshold is not None else gaussian_config.get("densify_grad_threshold")
    
    # 检查 gaussian-splatting 是否存在
    if not os.path.exists(gs_path):
        print(f"错误: gaussian-splatting 未找到: {gs_path}")
        print("请在配置文件中设置正确的路径")
        return None
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return None
    
    # 准备训练命令
    train_script = os.path.join(gs_path, "train.py")
    output_model_dir = os.path.join(data_dir, "output")
    log_dir = os.path.join(data_dir, "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 构建命令
    cmd = [
        "python", train_script,
        "-s", data_dir,
        "-m", output_model_dir,
        "--iterations", str(iterations),
        "--eval",
    ]
    
    # 添加可选参数
    if sh_degree is not None:
        cmd.extend(["--sh_degree", str(sh_degree)])
    
    if lambda_dssim is not None:
        cmd.extend(["--lambda_dssim", str(lambda_dssim)])
    
    if opacity_reset_interval is not None:
        cmd.extend(["--opacity_reset_interval", str(opacity_reset_interval)])
    
    if densify_grad_threshold is not None:
        cmd.extend(["--densify_grad_threshold", str(densify_grad_threshold)])
    
    # 显示训练信息
    print(f"\n{'='*80}")
    print("开始 3D Gaussian Splatting 训练")
    print(f"{'='*80}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_model_dir}")
    print(f"训练参数:")
    print(f"  - 迭代次数: {iterations}")
    if sh_degree is not None:
        print(f"  - SH 阶数: {sh_degree}")
    if lambda_dssim is not None:
        print(f"  - Lambda DSSIM: {lambda_dssim}")
    if opacity_reset_interval is not None:
        print(f"  - 透明度重置间隔: {opacity_reset_interval}")
    if densify_grad_threshold is not None:
        print(f"  - 密集化梯度阈值: {densify_grad_threshold}")
    print(f"日志文件: {log_file}")
    print(f"{'='*80}\n")
    
    # 保存训练配置
    config_file = os.path.join(log_dir, f"training_config_{timestamp}.txt")
    with open(config_file, "w") as f:
        f.write("3D Gaussian Splatting 训练配置\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"数据目录: {data_dir}\n")
        f.write(f"输出目录: {output_model_dir}\n")
        f.write(f"迭代次数: {iterations}\n")
        if sh_degree is not None:
            f.write(f"SH 阶数: {sh_degree}\n")
        if lambda_dssim is not None:
            f.write(f"Lambda DSSIM: {lambda_dssim}\n")
        if opacity_reset_interval is not None:
            f.write(f"透明度重置间隔: {opacity_reset_interval}\n")
        if densify_grad_threshold is not None:
            f.write(f"密集化梯度阈值: {densify_grad_threshold}\n")
        f.write(f"\n命令:\n{' '.join(cmd)}\n")
    
    try:
        # 运行训练
        with open(log_file, "w") as f:
            f.write(f"训练开始于 {timestamp}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                cwd=gs_path,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        print(f"\n{'='*80}")
        print("训练完成！")
        print(f"{'='*80}")
        print(f"模型保存到: {output_model_dir}")
        print(f"训练日志: {log_file}")
        print(f"\n查看模型:")
        print(f"  cd {gs_path}")
        print(f"  python viewer.py -m {output_model_dir}")
        print(f"{'='*80}\n")
        
        # 保存完成状态
        status_file = os.path.join(log_dir, "training_status.txt")
        with open(status_file, "w") as f:
            f.write("训练状态: 成功\n")
            f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {output_model_dir}\n")
            f.write(f"迭代次数: {iterations}\n")
            f.write(f"日志文件: {log_file}\n")
        
        return {
            "success": True,
            "model_path": output_model_dir,
            "iterations": iterations,
            "log_file": log_file,
            "config_file": config_file,
            "status_file": status_file,
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败: {e}")
        
        # 保存错误状态
        status_file = os.path.join(log_dir, "training_status.txt")
        with open(status_file, "w") as f:
            f.write("训练状态: 失败\n")
            f.write(f"失败时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误: {str(e)}\n")
            f.write(f"日志文件: {log_file}\n")
        
        return None
    except Exception as e:
        print(f"\n意外错误: {e}")
        return None


def compress_results(data_dir: str, mode: str = "model") -> str:
    """
    压缩训练结果
    
    Args:
        data_dir: 3dgs 数据目录
        mode: 压缩模式 ("model" 或 "full")
        
    Returns:
        压缩文件路径
    """
    import tarfile
    
    print(f"\n{'='*80}")
    print(f"压缩训练结果 ({mode} 模式)")
    print(f"{'='*80}")
    
    # 生成压缩文件名
    parent_dir = os.path.dirname(data_dir)
    run_name = os.path.basename(parent_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{run_name}_3dgs_{mode}_{timestamp}.tar.gz"
    archive_path = os.path.join(parent_dir, archive_name)
    
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            if mode == "model":
                # 仅压缩训练后的模型
                output_dir = os.path.join(data_dir, "output")
                if os.path.exists(output_dir):
                    tar.add(output_dir, arcname="output")
                    print(f"  ✓ 添加: output/ (训练模型)")
                else:
                    print(f"  ✗ 警告: output/ 目录不存在")
                
                # 添加训练日志和状态
                training_logs = os.path.join(data_dir, "training_logs")
                if os.path.exists(training_logs):
                    status_file = os.path.join(training_logs, "training_status.txt")
                    if os.path.exists(status_file):
                        tar.add(status_file, arcname="training_logs/training_status.txt")
                        print(f"  ✓ 添加: training_logs/training_status.txt")
                    
                    # 添加最新的配置文件
                    config_files = sorted([f for f in os.listdir(training_logs) if f.startswith("training_config_")])
                    if config_files:
                        latest_config = os.path.join(training_logs, config_files[-1])
                        tar.add(latest_config, arcname=f"training_logs/{config_files[-1]}")
                        print(f"  ✓ 添加: training_logs/{config_files[-1]}")
                
                # 添加元数据
                metadata_file = os.path.join(data_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    tar.add(metadata_file, arcname="metadata.json")
                    print(f"  ✓ 添加: metadata.json")
                
            else:  # full
                # 压缩整个 3dgs 目录
                tar.add(data_dir, arcname="3dgs")
                print(f"  ✓ 添加: 完整 3dgs/ 目录")
        
        # 显示压缩信息
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"\n{'='*80}")
        print(f"压缩完成！")
        print(f"{'='*80}")
        print(f"文件: {archive_path}")
        print(f"大小: {size_mb:.1f} MB")
        print(f"{'='*80}\n")
        
        return archive_path
        
    except Exception as e:
        print(f"\n✗ 压缩失败: {e}")
        return None


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 检查运行目录
    if not os.path.exists(args.run_dir):
        print(f"错误: 运行目录不存在: {args.run_dir}")
        sys.exit(1)
    
    # Stage 3 目录
    stage3_dir = os.path.join(args.run_dir, "stage3_viewcrafter")
    if not os.path.exists(stage3_dir):
        print(f"错误: Stage 3 目录不存在: {stage3_dir}")
        print("请确保已经运行完 Stage 1-3")
        sys.exit(1)
    
    # 检查必要文件
    required_files = [
        "cameras.npz",
        "reconstructed_images",
        "pointcloud_reconstructed.ply"
    ]
    
    for file in required_files:
        path = os.path.join(stage3_dir, file)
        if not os.path.exists(path):
            print(f"错误: 缺少必要文件: {path}")
            sys.exit(1)
    
    print("=" * 80)
    print("从 Stage 3 输出直接训练 3D-GS")
    print("=" * 80)
    print(f"运行目录: {args.run_dir}")
    print(f"Stage 3 目录: {stage3_dir}")
    print("=" * 80)
    
    # 创建新的 3dgs 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(args.run_dir, f"3dgs_retrain_{timestamp}")
    
    # 从 Stage 3 生成 COLMAP 格式
    gaussian_config = config.get("gaussian", {})
    max_points = gaussian_config.get("max_init_points", 200000)
    
    try:
        # 转换为绝对路径
        stage3_dir = os.path.abspath(stage3_dir)
        data_dir = os.path.abspath(data_dir)
        
        export_info = export_colmap_from_stage3(stage3_dir, data_dir, max_points)
    except Exception as e:
        import traceback
        print(f"\n错误: COLMAP 导出失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)
    
    # 开始训练
    result = train_3dgs(data_dir, config, args)
    
    if result and result.get("success"):
        print("\n✓ 训练成功完成")
        
        # 压缩结果
        if args.compress:
            archive_path = compress_results(data_dir, args.compress_mode)
            if archive_path:
                print(f"✓ 压缩文件已保存: {archive_path}")
    else:
        print("\n✗ 训练失败，请查看日志文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
