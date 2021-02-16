#!/usr/bin/python3

''' Task 2 -- Heat equation'''
import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt


# Stolen from Hermdog's solution to exercise 1
class BoundaryCondition:
    DIRCHLET = 1
    NEUMANN = 2
    
    def __init__(self, type, value):
        self.type = type
        self.value = value


def g(bc, u0, t):
    """
    Gets the value at one of the boundaries at time t

    Parameters:
        bc : BoundaryCondition (Dirchlet or Neumann)
        u0 : initial condition/value at the boundary
        t : time after initial tim 0 (t >= 0)
    Returns:
        u(x) at the boundary corresponding to the given parameters (x=0 or x=1)
    """

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
    Solve the 1D heat equation using forward Euler method

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k/(h**2)
    U = u0(x) # initial, t = 0

    solution_matrix = np.zeros((N, M))
    solution_matrix[0] = U
    
    for (i, ti) in enumerate(t[1:]):
        U[0] = g(bc1, U[0], ti)
        U[-1] = g(bc2, U[-1], ti)
        U[1:-1] = U[1:-1] + r * (U[:-2] -2*U[1:-1] +U[2:])
        solution_matrix[i] = U
    return x, U, solution_matrix


def backward_euler(bc1, bc2, M, N, t_end):
    """
    Solve the 1D heat equation using backward Euler method

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k/(h**2)
    U = u0(x) # initial, t = 0

    solution_matrix = np.zeros((N, M))
    solution_matrix[0] = U

    m = M-2
    diag = np.repeat(1+2*r, m)
    offdiag = np.repeat(-r, m-1)
    A = csr_matrix(diags([diag, offdiag, offdiag], [0,1,-1]))
    for (i, ti) in enumerate(t[1:]):
        b = U[1:-1]
        b[0] += r*g(bc1, U[0], ti)
        b[-1] += r*g(bc2, U[-1], ti)
        U[1:-1] = spsolve(A, b) # There is probably a solver made for sparse arrays which is better
        solution_matrix[i] = U
    return x, U, solution_matrix


def crank_nicolson(bc1, bc2, M, N, t_end):
    """
    Solve the 1D heat equation using backward Crank-Nicolson

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k/(h**2)
    U = u0(x) # initial, t = 0

    solution_matrix = np.zeros((N, M))
    solution_matrix[0] = U

    m = M-2
    diag = np.repeat(1+r, m)
    offdiag = np.repeat(-r/2, m-1)
    A = csc_matrix(diags([diag, offdiag, offdiag], [0,1,-1]))
    for (i, ti) in enumerate(t[1:]):
        b = (r/2)*U[:-2] + (1-r)*U[1:-1] + (r/2)*U[2:]
        b[0] += (r/2)*g(bc1, U[0], ti)
        b[-1] += (r/2)*g(bc2, U[-1], ti)
        U[1:-1] = spsolve(A, b) # There is probably a solver made for sparse arrays which is better
        solution_matrix[i] = U
    return x, U, solution_matrix


def test_method(method, M, N, t_end):
    """
    Do a testrun and plot results for a numerical solver of the heat equation

    Parameters:
        method : the function name of the method to be tested e.g. forward_euler
        M : number of spacial grid points
        N : number of time grid points
        t_end : end time
    """

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    x, U_final, solutions = method(bc1, bc2, M, N, t_end)
    
    sample_times = np.linspace(0, t_end, 5)
    for (i, t) in enumerate(sample_times):
        plt.plot(x, solutions[i], ".",label=f"t={t}")
    plt.title(method.__name__)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ## Test forward Euler ##
    test_method(forward_euler, 100, 100, 1)

    ## Test backward Euler
    test_method(backward_euler, 100, 100, 1)

    ## Test Crank-Nicolson
    test_method(crank_nicolson, 100, 100, 1)
