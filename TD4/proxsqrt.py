import numpy as np

def proxsqrt(x, g):
    """
    PROXSQRT Computation of the proximity operator of 
    f(x) = -g sqrt(x)
    where x and g are vectors of equal dimension
    """
    p = np.zeros(x.shape)
    
    for i in range(len(x)):
        r = np.roots([1, 0, -x[i], -g[i] / 2])
        r = np.real(r)
        u = r[r >= 0] ** 2
        fmin_idx = np.argmin((u - x[i]) ** 2 / 2 - g[i] * np.sqrt(u))
        p[i] = u[fmin_idx]
    
    return p.reshape(-1)