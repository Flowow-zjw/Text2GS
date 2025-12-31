"""
Text2GS Setup
"""

from setuptools import setup, find_packages

setup(
    name="text2gs",
    version="0.1.0",
    description="Text to 3D Gaussian Splatting",
    author="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "einops>=0.6.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.4.0",
        "opencv-python>=4.7.0",
        "scipy>=1.9.0",
        "trimesh>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "text2gs=text2gs.run:main",
        ],
    },
)
