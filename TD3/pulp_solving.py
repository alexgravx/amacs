# Pulp solving function for a problem in the following form
# minimize <C,X> st AX <= B and Aeq*X = Beq
# C is the cost matrix
# A, Aeq, B and Beq are constraints matrix

import pulp

def pulp_solve(C, A, Aeq, B, Beq, B_bound, N):
    """
    Résolution numérique du problème avec Pulp

    N est le nombre d'items (contraintes d'égalité)
    B est le nombre de boites -> on prend une borne sup avec heuristique
    """

    # Création du problème de minimisation
    prob = pulp.LpProblem("BinPacking_ILP", pulp.LpMinimize)

    # Définition des variables entières (par défaut >= 0)
    # On nomme x_vars les variables associées aux items et y_vars pour les variables associées aux boîtes.
    x_vars = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(B_bound * N)]
    y_vars = [pulp.LpVariable(f"y{i}", lowBound=0, cat="Integer") for i in range(B_bound)]

    # On regroupe toutes les variables dans une liste pour un accès facile
    new_X = x_vars + y_vars

    # Fonction objectif
    prob += pulp.lpDot(C, new_X)

    # Contraintes d'inégalité - B_val contraintes (une par boîte)
    for row in range(B_bound):
        constr_expr = 0
        for j in range(B_bound * (N + 1)):
            coeff = A[row, j]
            constr_expr += coeff * new_X[j]
        prob += (constr_expr <= B[row]), f"Inequality_constr_{row}"

    # Contraintes d'égalité - N contraintes (une par item)
    for i in range(N):
        constr_expr = 0
        for j in range(B_bound * (N + 1)):
            coeff = Aeq[i, j]
            constr_expr += coeff * new_X[j]
        prob += (constr_expr == Beq[i]), f"Equality_constr_{i}"

    print("Problème ILP construit. Résolution en cours...")
    prob.solve()
    print("Status :", pulp.LpStatus[prob.status])
    return x_vars, y_vars
