import numpy as np

def div(v, h):
    """
    Compute the divergence (adjoint of the spatial gradient).

    Parameters:
    v : ndarray (vertical gradient)
    h : ndarray (horizontal gradient)

    Returns:
    f : ndarray (divergence of v and h)
    """
    K, L = v.shape
    f = np.zeros((K + 1, L + 1))

    f[1:K+1, :L] = v
    f[:K, :L] -= v
    f[:K, 1:L+1] += h
    f[:K, :L] -= h

    return f
