import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from typing import Tuple, Dict

def mean_spectral_angle_distance(X: np.ndarray, Xhat: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes the Global Mean Spectral Angle Distance (MSAD) in degrees between 
    the original hyperspectral cube and the reconstructed cube.

    Parameters
    ----------
    X : np.ndarray
        Original ground truth HSI cube of shape (H, W, B).
    Xhat : np.ndarray
        Denoised/reconstructed HSI cube of shape (H, W, B).
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    float
        Average spectral angle distance in degrees.
    """
    H, W, B = X.shape
    Xv = X.reshape(-1, B)
    Yv = Xhat.reshape(-1, B)
    Xn = Xv / (np.linalg.norm(Xv, axis=1, keepdims=True) + eps)
    Yn = Yv / (np.linalg.norm(Yv, axis=1, keepdims=True) + eps)
    dot = np.sum(Xn * Yn, axis=1).clip(-1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    return float(np.mean(angles))

def detailed_evaluation(
    original: np.ndarray, 
    denoised: np.ndarray, 
    verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluates restoration performance using PSNR, SSIM across all spectral bands,
    and Global Mean Spectral Angle Distance (MSAD).

    Parameters
    ----------
    original : np.ndarray
        Ground truth HSI cube of shape (H, W, B).
    denoised : np.ndarray
        Denoised HSI cube of shape (H, W, B).
    verbose : bool
        If True, prints a formatted metric table to stdout.

    Returns
    -------
    Tuple[float, float, float]
        (mean_PSNR, mean_SSIM, global_MSAD)
    """
    H, W, B = original.shape
    psnr_list = []
    ssim_list = []

    for b in range(B):
        p = psnr(original[:, :, b], denoised[:, :, b], data_range=1.0)
        s = ssim(original[:, :, b], denoised[:, :, b], data_range=1.0)
        psnr_list.append(p)
        ssim_list.append(s)

    psnr_arr = np.array(psnr_list)
    ssim_arr = np.array(ssim_list)
    msad = mean_spectral_angle_distance(original, denoised)

    mean_p, med_p, max_p, min_p = np.mean(psnr_arr), np.median(psnr_arr), np.max(psnr_arr), np.min(psnr_arr)
    mean_s, med_s, max_s, min_s = np.mean(ssim_arr), np.median(ssim_arr), np.max(ssim_arr), np.min(ssim_arr)

    if verbose:
        print("\n" + "=" * 62)
        print(f" {'SPECTRA CLEAN - PERFORMANCE EVALUATION METRICS':^60}")
        print("=" * 62)
        print(f" {'Metric':<12} | {'Mean':<10} | {'Median':<10} | {'Max':<10} | {'Min':<10}")
        print("-" * 62)
        print(f" {'PSNR (dB)':<12} | {mean_p:<10.4f} | {med_p:<10.4f} | {max_p:<10.4f} | {min_p:<10.4f}")
        print(f" {'SSIM':<12} | {mean_s:<10.4f} | {med_s:<10.4f} | {max_s:<10.4f} | {min_s:<10.4f}")
        print("-" * 62)
        print(f" Global MSAD (deg): {msad:.4f}")
        print("=" * 62 + "\n")

    return mean_p, mean_s, msad
