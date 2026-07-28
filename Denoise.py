#!/usr/bin/env python3
"""
Legacy entry point for SpectraClean.

This script preserves backward compatibility with the original Denoise.py script
by delegating core functions to the modular `spectraclean` package.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from spectraclean import (
    load_hsi_file,
    normalize_hsi,
    add_mixed_noise,
    fast_rpca,
    pca_nlm_denoising,
    ultimate_pipeline,
    mean_spectral_angle_distance,
    detailed_evaluation,
    plot_comparison
)

if __name__ == "__main__":
    # --- [INPUT REQUIRED] SET HSI FILE PATH ---
    file_path = "Pavia_resized.npy"  
    
    # --- [INPUT REQUIRED] SET NOISE LEVELS ---
    gaussian_variance = 0.1   # Variance limit for Gaussian noise (G)
    salt_pepper_prob = 0.2    # Probability limit for Salt & Pepper noise (P)
    stripe_ratio = 0.2        # Fraction of spectral bands affected by stripes

    if not os.path.exists(file_path):
        print(f"[!] Target file '{file_path}' not found in current directory.")
        print("[+] Usage hint: Specify a valid .npy or .mat hyperspectral dataset path.")
        print("[+] Or run `python main.py --synthetic` to run a synthetic demo benchmark.")
        sys.exit(1)

    data = load_hsi_file(file_path).astype(np.float32)
    data_norm = normalize_hsi(data)

    noisy = add_mixed_noise(
        data_norm, 
        G=gaussian_variance, 
        P=salt_pepper_prob, 
        stripe_ratio=stripe_ratio, 
        seed=123
    )
    
    denoised = ultimate_pipeline(noisy, rank=75, n_pca=32)

    detailed_evaluation(data_norm, denoised)

    plot_comparison(data_norm, noisy, denoised, band_idx=25, show=True)
