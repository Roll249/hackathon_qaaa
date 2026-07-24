#!/usr/bin/env python3
"""Setup script for quantum-dengue-stpp package.

QC4SG 2026 Submission - Quantum-Enhanced Dengue Spatio-Temporal Point Process

Usage:
    pip install -e .
    python scripts/run_q_stpp_v17.py
    python reproduce_all.py
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="quantum-dengue-stpp",
    version="1.0.0",
    author="QC4SG 2026 Team",
    author_email="",
    description="Quantum-Enhanced Dengue Spatio-Temporal Point Process Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/quantum-dengue-stpp",
    project_urls={
        "Bug Reports": "https://github.com/your-org/quantum-dengue-stpp/issues",
        "Source": "https://github.com/your-org/quantum-dengue-stpp",
        "Documentation": "https://github.com/your-org/quantum-dengue-stpp#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pennylane>=0.38.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "networkx>=3.0,<4.0.0",
    ],
    extras_require={
        "gpu": [
            "pennylane-lightning[gpu] @ https://pennylaneai.github.io/Lightning-Wheels/pypi",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "nbformat>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-dengue-run=scripts.run_q_stpp_v17:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Quantum Computing Library :: PennyLane",
    ],
    keywords=[
        "quantum computing",
        "dengue prediction",
        "spatio-temporal point process",
        "QAOA",
        "PennyLane",
        "epidemiology",
        "machine learning",
    ],
    license="MIT",
    zip_safe=False,
)
