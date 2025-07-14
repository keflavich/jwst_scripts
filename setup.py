#!/usr/bin/env python3
"""
Setup script for jwst_rgb package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""

setup(
    name="jwst_rgb",
    version="0.1.0",
    author="Adam Ginsburg",
    author_email="adam.g.ginsburg@gmail.com",
    description="JWST RGB Image Processing Package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adamginsburg/jwst_scripts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="astronomy, jwst, rgb, image-processing, astrophysics",
    project_urls={
        "Bug Reports": "https://github.com/adamginsburg/jwst_scripts/issues",
        "Source": "https://github.com/adamginsburg/jwst_scripts",
    },
)