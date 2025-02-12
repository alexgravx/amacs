import numpy as np

def grad(f: np.ndarray):
    """
    Computes spatial gradients of image f
    
    Parameters:
    f : ndarray. Input image
    
    Returns:
    v : ndarray. Vertical gradient
    h : ndarray. Horizontal gradient
    """
    K, L = f.shape
    
    v = f[1:K, :L-1] - f[:K-1, :L-1]
    h = f[:K-1, 1:L] - f[:K-1, :L-1]
    
    return v, h