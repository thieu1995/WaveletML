#!/usr/bin/env python
# Created by "Thieu" at 13:29, 10/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import setuptools
import os
import re


with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()


def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'waveletml', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        init_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", init_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('README.md', encoding='utf-8') as f:
        res = f.read()
    return res


setuptools.setup(
    name="waveletml",
    version=get_version(),
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="WaveletML: A Scalable and Extensible Wavelet Neural Network Framework",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords = ["wavelet neural network", "wavelet transform", "WNN",
        "machine learning", "deep learning", "scikit-learn", "sklearn compatible",
        "wavelet", "pattern recognition", "morlet wavelet", "wavelet activation", "intelligent systems",
        "classification", "regression",  "signal processing",  "data science", "automated training",
        "PyTorch", "Classification", "Regression", "Supervised Learning", "Model Training",
        "Python Framework", "Neural Network", "Open-source Machine Learning",
        "Gradient Descent Optimization", "Metaheuristic Optimization", "Model Wrapping",
        "Benchmarking", "machine learning", "artificial intelligence", "metaheuristics",
        "metaheuristic optimization", "nature-inspired algorithms", "generalization",
        "optimization algorithms", "model selection", "Cross-validation"
        "Genetic algorithm (GA)", "Particle swarm optimization (PSO)", "Ant colony optimization (ACO)",
        "Differential evolution (DE)", "Simulated annealing", "Grey wolf optimizer (GWO)",
        "Whale Optimization Algorithm (WOA)", "automl", "parameter search", "mealpy", "search algorithm",
        "optimization framework", "global optimization", "local optimization",
        "Computational intelligence", "Robust optimization", "metaheuristic algorithms",
        "nature-inspired computing", "swarm-based computation", "gradient-free optimization"],
    url="https://github.com/thieu1995/WaveletML",
    project_urls={
        'Documentation': 'https://waveletml.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/WaveletML',
        'Bug Tracker': 'https://github.com/thieu1995/WaveletML/issues',
        'Change Log': 'https://github.com/thieu1995/WaveletML/blob/main/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=setuptools.find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": ["pytest==7.1.2", "pytest-cov==4.0.0", "flake8>=4.0.1"],
    },
    python_requires='>=3.8',
)
