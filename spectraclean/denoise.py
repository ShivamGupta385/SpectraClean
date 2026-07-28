import numpy as np
from sklearn.utils.extmath import randomized_svd
from skimage.restoration import denoise_nl_means, estimate_sigma

def fast_rpca(
    M: np.ndarray, 
    rank_est: int = 40, 
    lam: float = None, 
    mu: float = None, 
    max_iter: int = 100, 
    tol: float = 1e-6
) -> np.ndarray:
    """
    Fast Randomized Robust Principal Component Analysis (RPCA) using Inexact ALM.
    Decomposes a matrix M into M = L + S (Low-rank structural component + Sparse noise).

    Parameters
    ----------
    M : np.ndarray
        Input matrix of shape (pixels, bands).
    rank_est : int
        Estimated rank for randomized SVD computation.
    lam : float, optional
        Sparsity regularization parameter lambda (default: 1 / sqrt(max(m, n))).
    mu : float, optional
        Augmented Lagrangian multiplier parameter.
    max_iter : int
        Maximum number of ALM iterations.
    tol : float
        Convergence error tolerance threshold.

    Returns
    -------
    np.ndarray
        Low-rank matrix component L of shape (pixels, bands).
    """
    M = M.astype(np.float64)
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M.ravel(), np.inf) / lam)
    S = np.zeros_like(M)
    L = np.zeros_like(M)
    if mu is None:
        mu = 1.25 / np.linalg.norm(M, 2)
    rho = 1.5

    for it in range(max_iter):
        temp = M - S + (1.0 / mu) * Y
        U, s, Vt = randomized_svd(temp, n_components=rank_est, n_iter=5, random_state=42)
        s_thresh = np.maximum(s - 1.0 / mu, 0)
        rank = np.sum(s_thresh > 0)
        L = (U[:, :rank] * s_thresh[:rank]) @ Vt[:rank, :]

        temp = M - L + (1.0 / mu) * Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

        Z = M - L - S
        err = np.linalg.norm(Z, 'fro') / np.linalg.norm(M, 'fro')
        Y += mu * Z
        mu *= rho

        if err < tol:
            break
    return L

def pca_nlm_denoising(hsi_data: np.ndarray, n_components: int = 15) -> np.ndarray:
    """
    Denoises low-rank HSI data using Principal Component Analysis (PCA) projection
    followed by Non-Local Means (NLM) spatial filtering on principal components.

    Parameters
    ----------
    hsi_data : np.ndarray
        HSI data cube of shape (H, W, B).
    n_components : int
        Number of PCA eigen-images to retain and filter.

    Returns
    -------
    np.ndarray
        Spatial-spectrally denoised HSI data cube of shape (H, W, B).
    """
    H, W, B = hsi_data.shape
    X = hsi_data.reshape(-1, B)

    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    n_comp = min(n_components, B, X.shape[0])
    U, s, Vt = randomized_svd(X_centered, n_components=n_comp, random_state=42)

    eig_images_flat = X_centered @ Vt.T
    eig_images = eig_images_flat.reshape(H, W, n_comp)

    denoised_eig = np.zeros_like(eig_images)

    for k in range(n_comp):
        img = eig_images[:, :, k]
        sigma_est = np.mean(estimate_sigma(img))

        denoised_eig[:, :, k] = denoise_nl_means(
            img,
            h=0.6 * sigma_est,
            sigma=sigma_est,
            fast_mode=True,
            patch_size=5,
            patch_distance=7
        )

    denoised_flat = denoised_eig.reshape(-1, n_comp) @ Vt[:n_comp, :]
    denoised_hsi = denoised_flat + mean_vec

    return denoised_hsi.reshape(H, W, B)

def ultimate_pipeline(noisy: np.ndarray, rank: int = 40, n_pca: int = 16) -> np.ndarray:
    """
    Two-stage ultimate hyperspectral denoising pipeline:
    1. Fast RPCA for sparse noise removal and low-rank structural recovery.
    2. PCA dimension reduction with spatial NLM filtering for residual smoothing.

    Parameters
    ----------
    noisy : np.ndarray
        Noisy HSI data cube of shape (H, W, B).
    rank : int
        Estimated rank for RPCA decomposition.
    n_pca : int
        Number of PCA components for spatial NLM denoising.

    Returns
    -------
    np.ndarray
        Denoised clean HSI data cube of shape (H, W, B), clipped to [0, 1].
    """
    H, W, B = noisy.shape
    M = noisy.reshape(-1, B)
    
    # Stage 1: RPCA
    L_flat = fast_rpca(M, rank_est=rank, max_iter=100)
    low_rank_cube = L_flat.reshape(H, W, B)
    
    # Stage 2: PCA + NLM
    final_denoised = pca_nlm_denoising(low_rank_cube, n_components=n_pca)
    
    return np.clip(final_denoised, 0, 1)
