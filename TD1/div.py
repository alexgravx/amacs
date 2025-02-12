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
    f = np.zeros((K, L))

    f[1:, :] += v[:-1, :]
    f[:-1, :] -= v[:-1, :]
    f[:, 1:] += h[:, :-1]
    f[:, :-1] -= h[:, :-1]

    return f