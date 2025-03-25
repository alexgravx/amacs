import numpy as np
import time
from phasei import phasei
from phaseii import phaseii

def simplexe(A, b, c):
    """
    Algorithme du simplexe
    Permet de résoudre le problème
    max (cx) sous la contrainte Ax = b , avec b >= 0

    En sortie, elle retourne :
    zmax : la valeur du profit à l'optimum.
    PHIiter : le nombre d'itération dans la phase I.
    PHIIiter : le nombre d'itération dans la phase II.
    xbasic : Les valeurs des variables de base à l'optimum.
    ibasic : les indices des variables de base à l'optimum.

    Cette routine est le clone de 'linprog', développé par Jeff Stuart
    """
    # Dimensions de A
    m, n = A.shape
    
    # Vérifications des dimensions
    if b.shape != (m,):
        print("The dimensions of b do not match the dimensions of A.")
        return
    
    if np.min(b) < 0:
        print("The RHS vector b must be nonnegative.")
        return
    
    if c.shape != (n,):
        print("The dimensions of c do not match the dimensions of A.")
        return
    
    if np.linalg.matrix_rank(A) != m:
        print("A does not have full row rank.")
        return
    
    print("Everything good, starting calculus")
    # Variables initiales
    PHIiter = 0
    PHIIiter = 0
    tol = 1e-10
    xbasic = np.zeros(n)
    
    # Appel de la phase I
    wmax, ibasic, PHIiter = phasei(A, b)
    
    if wmax < -tol:
        t = time.time()
        print("The original LP is infeasible. Infeasibility was")
        print("detected during Phase I. The total number of phase")
        print("one iterations performed was:", PHIiter)
        return
    else:
        print("Phase I completed. Original LP is feasible.")
        print("The total number of Phase I iterations was:", PHIiter)
        print("Starting Phase II.")
        
        # Appel de la phase II
        zmax, xbasic, ibasic, ienter, PHIIiter, PCOL, OPTEST, CYCTEST = phaseii(A, b, c, ibasic)
        xbasic = xbasic.T
        
        if CYCTEST == 1:
            return
        
        if OPTEST == 0:
            print("The original LP is unbounded. An unbounded ray was")
            print("detected during Phase II. The output objective")
            print("value is for the last basic solution found.")
            print("The number of Phase II iterations was:", PHIIiter)
            print("Last objective value is", zmax)
            print("The last basic solution, xbasic is", xbasic)
            print("The column indices for the last basis:", ibasic)
            print("The index of the unbounded entering variable:", ienter)
            print("The unbounded ray column is:", PCOL)
        else:
            print("The original LP has an optimal solution.")
            print("The number of Phase II iterations was:", PHIIiter)
            print("The optimal objective value is", zmax)
            print("The indices for the basic columns:", ibasic)
            print("The optimal, basic solution is", xbasic)
    
    return zmax, PHIiter, PHIIiter, xbasic, ibasic
