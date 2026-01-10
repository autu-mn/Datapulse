"""
GitPulse Python Package Setup
"""

from setuptools import setup, find_packages
import os

# 读取 README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取 requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="gitpulse",
    version="1.0.0",
    author="GitPulse Team",
    author_email="your-email@example.com",  # 请修改为你的邮箱
    description="Multimodal Time Series Prediction for GitHub Project Health",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gitpulse",  # 如果有 GitHub 仓库
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gitpulse/issues",
        "Documentation": "https://huggingface.co/Patronum-ZJ/GitPulse",
        "Source Code": "https://github.com/yourusername/gitpulse",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Version Control :: Git",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
    },
    keywords="time-series, forecasting, github, transformer, multimodal, pytorch",
)





