#!/usr/bin/env python3
"""
Setup script for Cardiovascular Disease Prediction using Deep Learning
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cardiovascular-disease-prediction",
    version="1.0.0",
    author="V Sasikala, J. Arunarasi, S. Surya, N. Shivaanivarsha, Guru Raghavendra S, Gnanasudharsan A",
    author_email="sasikala.ece@sairam.edu.in",
    description="Deep Learning approach for Cardiovascular Disease Prediction using BDLSTM and CatBoost",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cardiovascular-disease-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/cardiovascular-disease-prediction/issues",
        "Documentation": "https://github.com/yourusername/cardiovascular-disease-prediction/docs",
        "Source Code": "https://github.com/yourusername/cardiovascular-disease-prediction",
        "Research Paper": "https://ieeexplore.ieee.org/document/10390290"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cvd-predict=main:main",
            "cvd-train=src.training:main",
            "cvd-evaluate=src.utils.evaluation_metrics:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yaml", "*.yml"],
        "data": ["*.csv"],
        "models": ["*.h5", "*.pkl"],
        "results": ["*.json", "*.png", "*.html"],
    },
    zip_safe=False,
    keywords=[
        "cardiovascular disease",
        "deep learning",
        "healthcare",
        "medical AI",
        "LSTM",
        "CatBoost",
        "machine learning",
        "prediction",
        "classification",
        "medical diagnosis",
        "artificial intelligence",
        "feature selection",
        "SHAP",
        "ensemble learning"
    ],
)
