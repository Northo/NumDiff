''' Task 2 -- Heat equation'''
import numpy as np
from scipy.sparse import diags
from matplotlib import pyplot as plt


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
        U[0] = bc1 # bc1 and bc2 can be functions of time, but currently only numbers 
        U[-1] = bc2 # so having this asign in the loop is not needed.
        U[1:-1] = U[1:-1] + r * (U[:-2] -2*U[1:-1] +U[2:])
    return x, U


def forward_euler_test():
    M = 100
    N = 10000
    # These should be the Dirchlet BC's corresponding to the given Neumann BC's
    bc1 = u0(0)
    bc2 = u0(1)
    x, U_0 = forward_euler(bc1, bc2, M, N, 0.0)
    x, U_1 = forward_euler(bc1, bc2, M, N, 0.01)
    x, U_2 = forward_euler(bc1, bc2, M, N, 0.02)
    x, U_3 = forward_euler(bc1, bc2, M, N, 0.03)
    x, U_4 = forward_euler(bc1, bc2, M, N, 0.1)
    # plot
    plt.plot(x, u0(x))
    plt.plot(x, U_0, ".")
    plt.plot(x, U_1, ".")
    plt.plot(x, U_2, ".")
    plt.plot(x, U_3, ".")
    plt.plot(x, U_4, ".")
    plt.show()


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
        b[0] += r*g0(ti)
        b[-1] += r*g1(ti)
        # solve AhU=b
    print(A.toarray())


def backwards_euler_test():
    M = 6
    N = 10
    t_end = 1
    backwards_euler(1,1, M, N, t_end)
    

if __name__ == "__main__":
    ## Test forward Euler ##
    forward_euler_test()

    ## Test Backwards Euler
    # backwards_euler_test()
    
