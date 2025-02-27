import numpy as np
import matplotlib.pyplot as plt
from div import div
from grad import grad
from TD1.projball2_enonce import projball2

def load_image(filename):
    return np.loadtxt(filename)

def display_image(image, title="Image"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()

def gradient_algorithm(z, nitm):
    K, L = z.shape
    xr = z.copy()
    cost = []
    for nit in range(nitm):
        v, h = grad(xr)
        xr = None  # TO BE COMPLETED
        cost.append((np.linalg.norm(v, 'fro')**2 + np.linalg.norm(h, 'fro')**2) / 2)
        print(f"{nit+1} : cost={cost[-1]}")
    return xr, cost

def projected_gradient_algorithm(z, nitm, rho, indI):
    K, L = z.shape
    xr = z.copy()
    cost = []
    for nit in range(nitm):
        v, h = grad(xr)
        xr = None  # TO BE COMPLETED
        xr = projball2(xr, z, rho, indI)
        cost.append((np.linalg.norm(v, 'fro')**2 + np.linalg.norm(h, 'fro')**2) / 2)
        print(f"{nit+1} : cost={cost[-1]}")
        if None:  # TO BE COMPLETED
            break
    return xr, cost

def projected_gradient_algorithm_constraint(z, nitm):
    xr = z.copy()
    for nit in range(nitm):
        break # TO BE COMPLETED
    return xr

def accelerated_algorithm(z, nitm, beta2):
    gamma3 = 1 / beta2
    zeta = 2.05
    xr = z.copy()
    y = xr.copy()
    cost = []
    for nit in range(nitm):
        v, h = grad(y)
        div_vh = div(v, h)
        x_next = y - gamma3 * div_vh
        y = x_next + zeta * (x_next - xr)
        xr = x_next
        cost.append((np.linalg.norm(v, 'fro')**2 + np.linalg.norm(h, 'fro')**2) / 2)
        print(f"{nit+1} : cost={cost[-1]}")
    return xr, cost

def main():
    # load image
    z = load_image("TD1/marie_degraded")
    
    # image size
    K, L = z.shape
    print(f"The image size is {K} x {L}")
    
    # display image
    display_image(z, "Marie Degraded")
    
    # indices corresponding to tearing
    indJ = None  # TO BE COMPLETED
    print(f"Tearing represents {100 * len(indJ) / (K * L)} % of the image")
    
    # indices of complementary area
    indI = None  # TO BE COMPLETED
    
    # q4 : gradient algorithm for minimizing g o L
    nitm = 15000 # maximum number of iterations
    beta = 8 # Lipshitz constant of the gradient
    gamma = 1.9 / beta # step-size of the algorithm
    xr, cost = gradient_algorithm(z, nitm, beta, gamma)
    display_image(xr, "Restored Image with Quadratic Cost")
    
    # q6-8 : projected gradient algorithm for minimizing g o L subject to constraint
    rho = 0.2 * np.sqrt(K * L)
    precc = 1e-7; # precision for stopping criterion
    xr, cost = projected_gradient_algorithm(z, nitm, beta, gamma, rho, indI)
    plt.figure(3)
    plt.subplot(121)
    display_image(xr, "Restored Image with Constraint")
    plt.subplot(122)
    plt.plot(cost)
    plt.title("Convergence Plot")
    plt.show()
    
    # q10 : projected gradient algorithm for minimizing smoothed total variation
    # subject to constraint
    
    eta = 5e-3 # smoothing parameter for total variation
    beta2 = None # TO BE COMPLETED # Lipshitz constant of the gradient
    gamma2 = 1.9 / beta2 # step-size of the algorithm
    xr = projected_gradient_algorithm_constraint(z, nitm, beta2, gamma2, eta)
    plt.figure(4)
    plt.subplot(121)
    display_image(xr, "Restored Image with Smoothed TV")
    plt.subplot(122)
    plt.plot(cost)
    plt.title("Convergence Plot")
    plt.show()
    
    # q11 : accelerated algorithm
    
    gamma3 = 1 / beta2 # step-size of the algorithm
    zeta = 2.05 # inertia parameter
    xr_acc, cost_acc = accelerated_algorithm(z, nitm, beta2)
    plt.figure(5)
    plt.subplot(121)
    plt.imshow(xr_acc, cmap='gray')
    plt.title('Restored Image with Smoothed TV (Accelerated)')
    plt.subplot(122)
    plt.plot(cost_acc)
    plt.title('Convergence Plot with Acceleration')
    plt.show()
    
if __name__ == "__main__":
    main()
