import numpy as np

def projball2(xr, z, rho, indI):
    """
    Projection of xr onto the L2 ball || x(indI) - z(indI) || <= rho.

    Parameters:
    xr : ndarray (vector to be projected)
    z : ndarray (center of the ball)
    rho : float (radius of the ball)
    indI : ndarray (indices of constrained components)

    Returns:
    p : ndarray (projected vector)
    """
    p = xr.copy()

    no = np.linalg.norm(xr[indI] - z[indI])

    if no > rho:
        p[indI] = None # TO BE COMPLETED

    return p
