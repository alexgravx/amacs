import numpy as np
import matplotlib.pyplot as plt

def ellipplot(P, gamma=1, ecolor='r', xc=None):
    """
    Plots an ellipsoid defined by the matrix P.

    Parameters:
    - P: (2, 2) matrix representing the ellipsoid's shape
    - gamma: Scaling factor (default 1)
    - ecolor: Color of the ellipsoid (default 'r' for red)
    - xc: Center of the ellipsoid (default is the origin)
    """
    if xc is None:
        xc = np.zeros(P.shape[0])
    
    P = P / gamma
    
    P = np.linalg.cholesky(P)
    
    theta = np.linspace(-np.pi, np.pi, 1000)
    z = np.array([np.cos(theta), np.sin(theta)])

    x = np.linalg.inv(P).dot(z)
    
    x = x + np.expand_dims(xc, axis=1)
    
    plt.fill(x[0, :], x[1, :], ecolor)
    plt.axis('equal')
    plt.show()

# Example usage:
P = np.array([[1, -1], [-1, 1]])  # Example matrix P
ellipplot(P, gamma=1, ecolor='g', xc=np.array([1, 1]))  # Plot with a green color and center at (1,1)
