import numpy as np

def phasei(A, b, tol=1e-7, ztol=1e-7):
    """
    Performs Phase I of the simplex method on the constraints Ax = b and x >= 0.
    This determines if a feasible point exists. If wmax < 0, the original LP is infeasible. 
    If wmax = 0, the original LP is feasible.
    
    Parameters:
    - A: (m x n) matrix representing the constraints of the linear program
    - b: (m,) vector representing the right-hand side of the constraints
    - tol: tolerance value to test feasibility (default 1e-7)
    - ztol: tolerance value for zero checks (default 1e-7)
    
    Returns:
    - wmax: Artificial objective value
    - ibasic: Indices of the basic variables at optimality
    - PHIiter: Number of Phase I iterations performed
    """
    m, n = A.shape
    A = np.hstack([A, np.eye(m)])
    PHIiter = 0
    X = np.zeros(n + m, dtype=np.float64)  # Ensure X is of float type
    J = np.hstack([np.zeros(n), np.ones(m)])
    c = -J
    K = np.arange(n + m)
    J = J.astype(bool)
    ibasic = K[J]
    inon = K[~J]
    B = np.eye(m)
    xbasic = np.array(b, dtype=np.float64)  # Ensure xbasic is of float type
    w = -np.sum(xbasic)
    X[ibasic] = b
    Cred = np.ones(m) @ A[:, inon]
    
    loop = True
    while loop:
        if np.max(Cred) > ztol:
            PHIiter += 1
            Maxcost = np.max(Cred)
            j = np.argmax(Cred)
            ienter = inon[j]
            
            # Solve for PCOL, check for zero in B (to avoid division by zero)
            try:
                PCOL = np.linalg.solve(B, A[:, ienter])
            except np.linalg.LinAlgError:
                print("Error: B is singular, cannot solve.")
                break
            
            J[ienter] = 1
            TESTROWS = np.where(PCOL > ztol)[0]
            TESTCOL = PCOL[TESTROWS]
            
            if np.any(TESTCOL == 0):
                print("Warning: Division by zero detected in TESTCOL.")
                break
            
            minrat = np.min(xbasic[TESTROWS] / TESTCOL)
            jmin = np.argmin(xbasic[TESTROWS] / TESTCOL)
            iexit = ibasic[TESTROWS[jmin]]
            J[iexit] = 0
            
            if minrat > 0:
                xbasic -= minrat * PCOL
            X[ibasic] = xbasic
            X[ienter] = minrat
            X[iexit] = 0
            w += Maxcost * minrat
            ibasic = K[J]
            inon = K[~J]
            B = A[:, ibasic]
            xbasic = X[ibasic]
            Cred = c[inon] - (c[ibasic] / B) @ A[:, inon]
        elif np.all(Cred <= ztol):
            loop = False
    
    wmax = -np.sum(X[n:])
    if wmax >= -tol:
        X = X[:n]
        last = ibasic[m-1]
        K = np.arange(last)
        ibasic = K[J[:last]]
        while last > n:
            J = J[:last]
            K = np.arange(last)
            inon = K[~J]
            B = A[:, ibasic]
            inon = inon[inon <= n]
            j = np.where(((np.zeros(m-1), 1) / B) @ A[:, inon])[0]
            ienter = inon[j[0]]
            J[ienter] = 1
            J[last] = 0
            ibasic = K[J]
            last = ibasic[m-1]
            PHIiter += 1
    
    return wmax, ibasic, PHIiter


# Example usage:
A = np.array([[1, -1], [1, 2], [3, 4]])  # Example constraint matrix
b = np.array([1, 2, 3])  # Example right-hand side vector

wmax, ibasic, PHIiter = phasei(A, b)
print(f"wmax: {wmax}, ibasic: {ibasic}, PHIiter: {PHIiter}")
