"""
Setup external submodules for Text2GS
"""

import os
import subprocess
import sys
from pathlib import Path


SUBMODULES = {
    "MVDiffusion": {
        "url": "https://github.com/Tangshitao/MVDiffusion.git",
        "path": "extern/MVDiffusion",
    },
    "ViewCrafter": {
        "url": "https://github.com/Drexubery/ViewCrafter.git",
        "path": "extern/ViewCrafter",
    },
    "dust3r": {
        "url": "https://github.com/naver/dust3r.git",
        "path": "extern/dust3r",
        "recursive": True,
    },
}


def run_command(cmd: list, cwd: str = None):
    """Run shell command"""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False
    return True


def clone_repo(name: str, info: dict, project_root: Path):
    """Clone a repository"""
    full_path = project_root / info["path"]
    
    if full_path.exists():
        print(f"  Already exists: {full_path}")
        return True
    
    os.makedirs(full_path.parent, exist_ok=True)
    
    cmd = ["git", "clone"]
    if info.get("recursive"):
        cmd.append("--recursive")
    cmd.extend([info["url"], str(full_path)])
    
    return run_command(cmd)


def main():
    print("=" * 60)
    print("Text2GS Submodule Setup")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    for name, info in SUBMODULES.items():
        print(f"\n[{name}]")
        clone_repo(name, info, project_root)
    
    # Install DUSt3R requirements
    dust3r_path = project_root / "extern/dust3r"
    if dust3r_path.exists():
        print("\n[Installing DUSt3R requirements]")
        req_file = dust3r_path / "requirements.txt"
        if req_file.exists():
            run_command([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    
    # Check status
    print("\nSubmodule status:")
    for name, info in SUBMODULES.items():
        full_path = project_root / info["path"]
        status = "✓" if full_path.exists() else "✗"
        print(f"  {status} {name}: {full_path}")


if __name__ == "__main__":
    main()
