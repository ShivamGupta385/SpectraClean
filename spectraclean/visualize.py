import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_comparison(
    original: np.ndarray, 
    noisy: np.ndarray, 
    denoised: np.ndarray, 
    band_idx: int = 25,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots a side-by-side comparison of a single spectral band across
    Original, Noisy, and Denoised hyperspectral cubes.

    Parameters
    ----------
    original : np.ndarray
        Original ground truth HSI cube of shape (H, W, B).
    noisy : np.ndarray
        Corrupted noisy HSI cube of shape (H, W, B).
    denoised : np.ndarray
        Restored HSI cube of shape (H, W, B).
    band_idx : int
        Spectral band index to display.
    save_path : str, optional
        Path to save the resulting figure image (e.g., 'comparison.png').
    show : bool
        Whether to call plt.show().
    """
    B = original.shape[2]
    band = max(0, min(band_idx, B - 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(original[:, :, band], cmap='viridis')
    axes[0].set_title(f"Original (Band {band})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(noisy[:, :, band], cmap='viridis')
    axes[1].set_title(f"Noisy (Band {band})", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(denoised[:, :, band], cmap='viridis')
    axes[2].set_title(f"Denoised (Band {band})", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("SpectraClean Hyperspectral Band Restoration Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[+] Comparison plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_spectral_signature(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    pixel_coord: tuple = (10, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots the spectral reflectance curves across all bands for a single spatial pixel (row, col).

    Parameters
    ----------
    original : np.ndarray
        Original HSI cube (H, W, B).
    noisy : np.ndarray
        Noisy HSI cube (H, W, B).
    denoised : np.ndarray
        Denoised HSI cube (H, W, B).
    pixel_coord : tuple
        Spatial coordinate (r, c).
    save_path : str, optional
        File path to save the output plot.
    show : bool
        Whether to display the plot window.
    """
    r, c = pixel_coord
    bands = np.arange(original.shape[2])

    plt.figure(figsize=(10, 5))
    plt.plot(bands, original[r, c, :], label='Original', color='green', linewidth=2)
    plt.plot(bands, noisy[r, c, :], label='Noisy', color='red', alpha=0.5, linestyle='--')
    plt.plot(bands, denoised[r, c, :], label='Denoised', color='blue', linewidth=1.5)
    
    plt.title(f"Spectral Signature Comparison at Pixel ({r}, {c})", fontsize=12, fontweight='bold')
    plt.xlabel("Spectral Band Index", fontsize=11)
    plt.ylabel("Reflectance Intensity", fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[+] Spectral signature plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
