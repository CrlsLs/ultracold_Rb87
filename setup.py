# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:31:49 2026

@author: clher
"""

# setup.py - Save in ultracold_atoms_project/

from setuptools import setup, find_packages

setup(
    name="ultracold",
    version="0.1.0",
    author="Carlos Luis H.C.",
    description="Ultracold atoms experiment analysis and simulation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
    ],
)