''' Task 2 -- Heat equation'''
import numpy as np
from scipy.sparse import diags
from matplotlib import pyplot as plt


# Stolen from Hermdog's solution to exercise 1
class BoundaryCondition:
    DIRCHLET = 1
    NEUMANN = 2
    
    def __init__(self, type, value):
        self.type = type
        self.value = value


def g(bc, u0, t):
    """ Boundary condition """
    if bc.type == BoundaryCondition.DIRCHLET:
        return bc.value
    elif bc.type == BoundaryCondition.NEUMANN:
        return u0 + bc.value*t
    else:
        raise("unknown boundary condition type")


def u0(x):
    """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """
    return 2*np.pi*x - np.sin(2*np.pi*x)


def forward_euler(bc1, bc2, M, N, t_end):
    """
    Solve pde using forward Euler method

    Parameters:
        M : Number of spacial grid points
        N : Number of time grid points
    Returns:
        x : spatial grid
        U : solution of the heat equation at time t_end
    """
    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k/(h**2)
    U = u0(x) # initial, t = 0
    
    for ti in t[1:]:
        U[0] = g(bc1, U[0], ti)
        U[-1] = g(bc2, U[-1], ti)
        U[1:-1] = U[1:-1] + r * (U[:-2] -2*U[1:-1] +U[2:])
    return x, U


def backwards_euler(bc1, bc2, M, N, t_end):
    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k/(h**2)
    U = u0(x) # initial, t = 0

    m = M-2
    diag = np.repeat(1+2*r, m)
    offdiag = np.repeat(-r, m-1)
    A = diags([diag, offdiag, offdiag], [0,1,-1])
    for ti in t[1:]:
        b = U[1:-1]
        b[0] += r*g(bc1, U[0], ti)
        b[-1] += r*g(bc2, U[-1], ti)
        # solve AhU=b
        U[1:-1] = np.linalg.solve(A.toarray(), b)
    return x, U


def test_method(method, M, N):
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    x, U_0 = method(bc1, bc2, M, N, 0.0)
    x, U_1 = method(bc1, bc2, M, N, 0.01)
    x, U_2 = method(bc1, bc2, M, N, 0.02)
    x, U_3 = method(bc1, bc2, M, N, 0.03)
    x, U_4 = method(bc1, bc2, M, N, 0.1)
    # plot
    plt.plot(x, u0(x))
    plt.plot(x, U_0, ".")
    plt.plot(x, U_1, ".")
    plt.plot(x, U_2, ".")
    plt.plot(x, U_3, ".")
    plt.plot(x, U_4, ".")
    plt.show()
    

if __name__ == "__main__":
    ## Test forward Euler ##
    test_method(forward_euler, 100, 10000)

    ## Test Backwards Euler
    test_method(backwards_euler, 100, 100)
