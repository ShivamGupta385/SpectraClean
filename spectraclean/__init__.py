"""
SpectraClean - Hyperspectral Image (HSI) Mixed Noise Denoising Package

A robust, modular Python library for simulating complex mixed noise in Hyperspectral Images
and restoring spatial-spectral quality using Fast RPCA and PCA-NLM filtering.
"""

from .io import load_hsi_file, normalize_hsi
from .noise import add_mixed_noise
from .denoise import fast_rpca, pca_nlm_denoising, ultimate_pipeline
from .metrics import mean_spectral_angle_distance, detailed_evaluation
from .visualize import plot_comparison, plot_spectral_signature

__version__ = "1.0.0"
__author__ = "SpectraClean Contributors"

__all__ = [
    "load_hsi_file",
    "normalize_hsi",
    "add_mixed_noise",
    "fast_rpca",
    "pca_nlm_denoising",
    "ultimate_pipeline",
    "mean_spectral_angle_distance",
    "detailed_evaluation",
    "plot_comparison",
    "plot_spectral_signature"
]
