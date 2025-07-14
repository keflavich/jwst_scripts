"""
JWST RGB Image Processing Package

This package provides functionality for creating RGB images from JWST data
with support for transparency, contour overlays, and metadata embedding.
"""

from .save_rgb import save_rgb

__version__ = "0.1.0"
__author__ = "Adam Ginsburg"
__all__ = ["save_rgb"]