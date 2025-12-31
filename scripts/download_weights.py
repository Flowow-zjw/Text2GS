"""
Download model weights for Text2GS
"""

import os
import sys
import urllib.request
from pathlib import Path


WEIGHTS = {
    "dust3r": {
        "url": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "path": "checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "size": "~1GB",
    },
    "viewcrafter_sparse": {
        "url": "https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt",
        "path": "checkpoints/viewcrafter/model_sparse.ckpt",
        "size": "~10GB",
        "note": "Sparse view model - better for multi-view input",
    },
    "viewcrafter_single": {
        "url": "https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt",
        "path": "checkpoints/viewcrafter/model.ckpt",
        "size": "~10GB",
        "note": "Single view model - optional",
        "optional": True,
    },
    "mvdiffusion": {
        "url": None,  # Manual download required
        "path": "checkpoints/mvdiffusion/pano.ckpt",
        "size": "~5GB",
        "note": "Download from https://github.com/Tangshitao/MVDiffusion/releases",
    },
}


def download_file(url: str, path: str, desc: str = ""):
    """Download file with progress"""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Path: {path}")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, path, progress_hook)
    print("\n  Done!")


def main():
    print("=" * 60)
    print("Text2GS Weight Downloader")
    print("=" * 60)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    for name, info in WEIGHTS.items():
        print(f"\n[{name}]")
        
        # Skip optional weights unless explicitly requested
        if info.get("optional", False):
            print(f"  Optional - skipping (use --all to download)")
            continue
        
        full_path = project_root / info["path"]
        
        if full_path.exists():
            print(f"  Already exists: {full_path}")
            continue
        
        if info["url"] is None:
            print(f"  Manual download required!")
            print(f"  Size: {info['size']}")
            print(f"  Note: {info.get('note', '')}")
            print(f"  Save to: {full_path}")
            continue
        
        try:
            download_file(info["url"], str(full_path), name)
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Please download manually from: {info['url']}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Check status
    print("\nWeight status:")
    for name, info in WEIGHTS.items():
        full_path = project_root / info["path"]
        status = "✓" if full_path.exists() else "✗"
        print(f"  {status} {name}: {full_path}")


if __name__ == "__main__":
    main()
