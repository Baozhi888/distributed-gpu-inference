#!/usr/bin/env python3
"""
GPU Worker 安装脚本
支持 pip install 安装
"""
from setuptools import setup, find_packages

setup(
    name="gpu-worker",
    version="1.0.0",
    description="分布式GPU推理 Worker 节点",
    author="GPU Sharing Project",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "cuda": ["bitsandbytes>=0.41.0"],
    },
    entry_points={
        "console_scripts": [
            "gpu-worker=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
