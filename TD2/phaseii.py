import numpy as np
import matplotlib.pyplot as plt

def phaseii(A, b, c, ibasic):
    # PHASEII effectue la phase II de la méthode du simplexe en commençant avec les
    # colonnes de base spécifiées par le vecteur ibasic.
    
    m, n = A.shape
    PCOL = []
    ienter = []
    iter = 0
    cycle = 0
    CYCTEST = 0
    X = np.zeros(n)
    J = np.zeros(n)
    J[ibasic] = 1
    K = np.arange(n)
    inon = K[J == 0]
    B = A[:, ibasic]
    xbasic = np.linalg.solve(B, b)
    z = np.dot(c[ibasic], xbasic)
    
    if m < n:
        X[ibasic] = xbasic
        Cred = c[inon] - np.dot(c[ibasic] / B, A[:, inon])
        OPTEST = 1
        loop = 1
        while loop == 1:
            if np.max(Cred) > 0:
                iter += 1
                Maxcost = np.max(Cred)
                j = np.argmax(Cred)
                ienter = inon[j]
                PCOL = np.linalg.solve(B, A[:, ienter])
                
                if np.all(PCOL <= 0):
                    OPTEST = 0
                    loop = 0
                else:
                    J[ienter] = 1
                    TESTROWS = np.where(PCOL > 0)[0]
                    TESTCOL = PCOL[TESTROWS]
                    minrat = np.min(xbasic[TESTROWS] / TESTCOL)
                    if minrat <= 0:
                        cycle += 1
                        if cycle > m:
                            print('Algorithm terminated due to excessive cycling.')
                            print('Restart algorithm from phase II using a perturbed')
                            print(' RHS vector b and the current basis.')
                            print(ibasic)
                            CYCTEST = 1
                            break
                    else:
                        cycle = 0
                    
                    j = np.argmin(xbasic[TESTROWS] / TESTCOL)
                    iexit = ibasic[TESTROWS[j]]
                    J[iexit] = 0
                    xbasic -= minrat * PCOL
                    X[ibasic] = xbasic
                    X[ienter] = minrat
                    X[iexit] = 0
                    z += Maxcost * minrat
                    J = J.astype(bool)
                    ibasic = K[J]
                    inon = K[J == 0]
                    B = A[:, ibasic]
                    xbasic = X[ibasic]
                    plt.plot(xbasic[0], xbasic[1])
                    Cred = c[inon] - np.dot(c[ibasic] / B, A[:, inon])
            else:
                loop = 0
    
    return z, xbasic, ibasic, ienter, iter, PCOL, OPTEST, CYCTEST
