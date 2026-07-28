#!/usr/bin/env python3
"""
SpectraClean Quickstart Example

This script runs a complete synthetic benchmark pipeline without requiring
external dataset files. It generates a synthetic 3D hyperspectral cube,
injects mixed noise, runs RPCA + PCA-NLM denoising, and outputs metrics.
"""

import sys
import os

# Allow importing spectraclean package from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectraclean import (
    add_mixed_noise,
    ultimate_pipeline,
    detailed_evaluation,
    plot_comparison
)
from main import create_synthetic_hsi

def run_quickstart():
    print("==================================================")
    print("      SpectraClean Synthetic Quickstart Demo      ")
    print("==================================================")

    # 1. Generate Synthetic Data Cube (64 x 64 x 32)
    print("\n1. Generating synthetic HSI data cube (64x64x32)...")
    clean_hsi = create_synthetic_hsi(shape=(64, 64, 32), seed=42)

    # 2. Inject Mixed Noise
    print("2. Injecting mixed noise (Gaussian G=0.08, S&P P=0.15, Stripes=0.15)...")
    noisy_hsi = add_mixed_noise(
        clean_hsi, 
        G=0.08, 
        P=0.15, 
        stripe_ratio=0.15, 
        seed=100
    )

    # 3. Apply Denoising Pipeline
    print("3. Executing two-stage RPCA + PCA-NLM restoration pipeline...")
    denoised_hsi = ultimate_pipeline(noisy_hsi, rank=40, n_pca=16)

    # 4. Compute Metrics
    print("4. Evaluating quantitative performance metrics...")
    detailed_evaluation(clean_hsi, denoised_hsi, verbose=True)

    # 5. Save Output Visualization
    output_img = "quickstart_results.png"
    plot_comparison(clean_hsi, noisy_hsi, denoised_hsi, band_idx=15, save_path=output_img, show=False)
    print(f"\n[+] Demo complete! Saved comparison visualization to: {output_img}")

if __name__ == "__main__":
    run_quickstart()
