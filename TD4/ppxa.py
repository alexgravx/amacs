import numpy as np
from proxsqrt import proxsqrt

def PPXA(f, L, S, R, g=100, la=1, om=3/4, nitm=100000, prec=1e-7):
    """
    y = PPXA(f,L,S,R) solves the following optimization problem:
        
        max f'*sqrt(L y)
        subject to: y >= 0 ; y(1) = S ; y(end) = R

    where f is a vector, L is a linear operator, S is the initial
    value and R is the final value.
    Note that if no constraint is set on the end value, R should
    be set to [].

    y = PPXA(f,L,S,R,g,la,om,nitm,prec) solves the above problem
    and additionally allow for customization of the parameters of
    the algorithm:
        g    : stepsize in the prox of the cost function
        la   : relaxation parameter, 0 < la < 2
        om   : averaging parameter, 0 < om < 1
        nitm : maximum iteration number
        prec : precision accuracy

    PPXA+ algorithm designed by J.-C. Pesquet and N. Pustelnik 
    """
    K = len(f) + 1
    t1 = np.zeros(K - 1)
    t2 = np.zeros(K)
    p2 = np.zeros(K)
    y = np.zeros(K)
    
    Q = np.linalg.inv(om * L.T @ L + (1 - om) * np.eye(K))
    oldcost = -1
    
    for nit in range(1, nitm + 1):
        p1 = proxsqrt(t1, g * f)
        
        p2[0] = S
        print(p2)
        if R:
            p2[K - 1] = R
            p2[1:K - 1] = np.maximum(t2[1:K - 1], 0)
        else:
            p2[1:K] = np.maximum(t2[1:K], 0)
        
        c = Q @ (om * L.T @ p1 + (1 - om) * p2)
        
        t1 = t1 + la * (L @ (2 * c - y) - p1)
        t2 = t2 + la * (2 * c - y - p2)
        
        y = y + la * (c - y)
        
        newcost = f.T @ np.sqrt(L @ p2)
        
        if nit % 100 == 0:
            print(f'it = {nit} cost = {newcost}')
        
        if abs(oldcost - newcost) < prec:
            break
        
        oldcost = newcost
    
    return p2