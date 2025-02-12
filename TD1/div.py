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

    n = np.arange(K)
    m = np.arange(L)

    f[np.ix_(n + 1, m)] += v
    f[np.ix_(n, m)] -= v
    f[np.ix_(n, m + 1)] += h
    f[np.ix_(n, m)] -= h

    return f[:-1, :-1]