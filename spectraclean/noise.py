import numpy as np

def add_mixed_noise(
    image: np.ndarray, 
    G: float = 0.1, 
    P: float = 0.2, 
    stripe_ratio: float = 0.2, 
    seed: int = 0
) -> np.ndarray:
    """
    Simulates real-world sensor degradation on an HSI data cube by injecting
    a mixture of Gaussian noise, Impulse (Salt & Pepper) noise, and Structural Stripes.

    Parameters
    ----------
    image : np.ndarray
        Clean normalized HSI image cube of shape (H, W, B) in range [0, 1].
    G : float
        Upper bound for Gaussian noise variance across bands.
    P : float
        Upper bound for Salt & Pepper impulse noise probability across bands.
    stripe_ratio : float
        Fraction of total spectral bands affected by random column stripes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy HSI image cube clipped to range [0, 1].
    """
    rng = np.random.default_rng(seed)
    img = np.clip(image, 0, 1).astype(np.float32)
    H, W, B = img.shape
    noisy = img.copy()

    def add_sp_noise_band(band: np.ndarray, p: float) -> np.ndarray:
        if p <= 0:
            return band
        mask = rng.choice([0, 1, 2], size=band.shape, p=[p / 2, 1 - p, p / 2])
        out = band.copy()
        out[mask == 0] = 0.0
        out[mask == 2] = 1.0
        return out

    # Add Gaussian and Salt & Pepper noise to each spectral band
    for b in range(B):
        sigma = np.sqrt(rng.uniform(0, G))
        band = noisy[:, :, b] + rng.normal(0, sigma, (H, W)).astype(np.float32)
        band = add_sp_noise_band(band, rng.uniform(0, P))
        noisy[:, :, b] = band

    # Add vertical stripe artifacts to a subset of bands
    stripe_bands = rng.choice(B, size=max(1, int(stripe_ratio * B)), replace=False)
    for b in stripe_bands:
        for _ in range(rng.integers(2, 8)):
            col = rng.integers(0, W)
            width = rng.integers(1, 3)
            noisy[:, col:col + width, b] = np.clip(
                noisy[:, col:col + width, b] + rng.uniform(0.08, 0.2), 0, 1
            )

    return np.clip(noisy, 0, 1)
