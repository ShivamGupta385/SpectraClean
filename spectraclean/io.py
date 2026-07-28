import os
import numpy as np
from scipy.io import loadmat

def load_hsi_file(path: str) -> np.ndarray:
    """
    Loads a Hyperspectral Image (HSI) cube from a .mat or .npy file.
    
    If a .mat file is supplied, it extracts the 3D numpy array and 
    caches it as a .npy file for faster subsequent loading.

    Parameters
    ----------
    path : str
        Path to the HSI data file (.npy or .mat).

    Returns
    -------
    np.ndarray
        3D hyperspectral data cube of shape (H, W, B).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(path)[0]
    npy_path = base + ".npy"

    if ext == ".npy":
        return np.load(path)
    elif ext == ".mat":
        mat_data = loadmat(path)
        data = None
        for k, v in mat_data.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                data = v
                break
        if data is None:
            raise ValueError(f"No 3D hyperspectral array found in .mat file '{path}'!")
        
        # Save converted cache
        np.save(npy_path, data.astype(np.float32))
        return np.load(npy_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .mat or .npy file.")

def normalize_hsi(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalizes hyperspectral image data cube to [0.0, 1.0] range.

    Parameters
    ----------
    data : np.ndarray
        Raw HSI data cube of shape (H, W, B).
    eps : float
        Small float epsilon for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized HSI data cube in float32 format within range [0, 1].
    """
    data_float = data.astype(np.float32)
    d_min = data_float.min()
    d_max = data_float.max()
    return (data_float - d_min) / (d_max - d_min + eps)
