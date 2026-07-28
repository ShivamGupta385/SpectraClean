#!/usr/bin/env python3
"""
SpectraClean - Main CLI Execution Script

Usage:
    python main.py --file Pavia_resized.npy --g-noise 0.1 --sp-noise 0.2 --stripes 0.2
    python main.py --synthetic --save-plot comparison.png
"""

import argparse
import os
import sys
import numpy as np

from spectraclean import (
    load_hsi_file,
    normalize_hsi,
    add_mixed_noise,
    ultimate_pipeline,
    detailed_evaluation,
    plot_comparison,
    plot_spectral_signature
)

def create_synthetic_hsi(shape=(64, 64, 30), seed=42) -> np.ndarray:
    """Generates a synthetic smooth 3D hyperspectral data cube for quick testing."""
    rng = np.random.default_rng(seed)
    H, W, B = shape
    x = np.linspace(0, np.pi * 2, W)
    y = np.linspace(0, np.pi * 2, H)
    xx, yy = np.meshgrid(x, y)
    
    cube = np.zeros(shape, dtype=np.float32)
    for b in range(B):
        phase = (b / B) * np.pi
        spatial_pattern = np.sin(xx + phase) * np.cos(yy + phase)
        cube[:, :, b] = (spatial_pattern - spatial_pattern.min()) / (spatial_pattern.max() - spatial_pattern.min() + 1e-8)
    
    return cube

def parse_args():
    parser = argparse.ArgumentParser(
        description="SpectraClean: Hyperspectral Image (HSI) Mixed Noise Denoising Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-f", "--file", type=str, default="Pavia_resized.npy",
        help="Path to HSI dataset file (.npy or .mat)."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic benchmark HSI data cube instead of loading a file."
    )
    parser.add_argument(
        "-g", "--g-noise", type=float, default=0.1,
        help="Upper limit for Gaussian noise variance G."
    )
    parser.add_argument(
        "-p", "--sp-noise", type=float, default=0.2,
        help="Upper limit for Salt & Pepper impulse noise probability P."
    )
    parser.add_argument(
        "-s", "--stripes", type=float, default=0.2,
        help="Fraction of spectral bands affected by vertical stripe artifacts."
    )
    parser.add_argument(
        "-r", "--rank", type=int, default=75,
        help="Estimated matrix rank for RPCA decomposition."
    )
    parser.add_argument(
        "-n", "--n-pca", type=int, default=32,
        help="Number of PCA components for spatial NLM filtering."
    )
    parser.add_argument(
        "-b", "--band", type=int, default=25,
        help="Band index to visualize in comparison plot."
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="File path to save the visual comparison plot (e.g. results.png)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable interactive plot display."
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for noise generation reproducibility."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(f" {'SPECTRA CLEAN - HYPERSPECTRAL DENOISING PIPELINE':^58}")
    print("=" * 60)

    # 1. Load Data
    if args.synthetic:
        print("[+] Generating synthetic benchmark HSI data cube (64x64x30)...")
        data_norm = create_synthetic_hsi()
    else:
        if not os.path.exists(args.file):
            print(f"[!] Warning: File '{args.file}' not found.")
            print("[+] Falling back to synthetic HSI data cube for demonstration.")
            data_norm = create_synthetic_hsi()
        else:
            print(f"[+] Loading HSI data from: {args.file}")
            raw_data = load_hsi_file(args.file)
            data_norm = normalize_hsi(raw_data)

    print(f"[+] HSI Data Cube Dimensions: H={data_norm.shape[0]}, W={data_norm.shape[1]}, Bands={data_norm.shape[2]}")

    # 2. Inject Mixed Noise
    print(f"[+] Simulating mixed noise: Gaussian Var={args.g-noise}, S&P Prob={args.sp-noise}, Stripe Ratio={args.stripes}")
    noisy = add_mixed_noise(
        data_norm,
        G=args.g_noise,
        P=args.sp_noise,
        stripe_ratio=args.stripes,
        seed=args.seed
    )

    # 3. Denoising Pipeline
    print(f"[+] Running Ultimate Pipeline (RPCA rank={args.rank}, PCA components={args.n_pca})...")
    denoised = ultimate_pipeline(noisy, rank=args.rank, n_pca=args.n_pca)

    # 4. Quantitative Evaluation
    detailed_evaluation(data_norm, denoised, verbose=True)

    # 5. Visualization
    show_plot = not args.no_plot
    if show_plot or args.save_plot:
        plot_comparison(
            data_norm, 
            noisy, 
            denoised, 
            band_idx=args.band, 
            save_path=args.save_plot, 
            show=show_plot
        )

if __name__ == "__main__":
    main()
