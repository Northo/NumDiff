#!/usr/bin/python3

""" Task 2 -- Heat equation """
import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from matplotlib import pyplot as plt


# Hermdog's BC class from exercise 1
class BoundaryCondition:
    DIRCHLET = 1
    NEUMANN = 2

    def __init__(self, type, value):
        self.type = type
        if callable(value):
            self.value = value
        else:
            self.value = lambda t: value


def initial(x):
    """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

    return 2 * np.pi * x - np.sin(2 * np.pi * x)


def forward_euler(bc1, bc2, M, N, t_end, u0=initial, log=True):
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
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    for (i, ti) in enumerate(t[1:]):
        if bc1.type == BoundaryCondition.DIRCHLET:
            U[0] = bc1.value(ti)
        elif bc1.type == BoundaryCondition.NEUMANN:
            U[0] += r * (U[1] - U[0] - 2 * h * bc1.value(ti))
        else:
            raise ("Unsupported boundary condition type")
        if bc2.type == BoundaryCondition.DIRCHLET:
            U[-1] = bc2.value(ti)
        elif bc2.type == BoundaryCondition.NEUMANN:
            U[-1] += r * (U[-2] - U[-1] + 2 * h * bc1.value(ti))
        else:
            raise ("Unsupported boundary condition type")
        U[1:-1] = U[1:-1] + r * (U[:-2] - 2 * U[1:-1] + U[2:])
        if log:
            solution_matrix[i+1] = U
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def backward_euler(bc1, bc2, M, N, t_end, u0=initial, log=True):
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
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
        # neumann-neumann
        diag = np.repeat(1 + 2 * r, M)
        offdiag_upper = np.repeat(-r, M - 1)
        offdiag_lower = np.repeat(-r, M - 1)
        diag[0] = 1 - r
        offdiag_upper[0] = r
        diag[-1] = 1 + r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U
            b[0] -= r * 2 * r * h * bc1.value(ti)
            b[-1] += r * 2 * r * h * bc2.value(ti)
            U = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # dirchlet-dirchlet
        m = M - 2
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[1:-1]
            b[0] += r * bc1.value(ti)
            b[-1] += r * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET and bc2.type == BoundaryCondition.NEUMANN
    ):
        # dirchlet-neumann
        m = M - 1
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        diag[-1] = 1 + r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[1:]
            b[0] += r * bc1.value(ti)
            b[-1] += r * 2 * r * h * bc2.value(ti)
            U[1:] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # neumann-dirchlet
        m = M - 1
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        diag[0] = 1 - r
        offdiag_upper[0] = r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[:-1]
            b[0] -= r * 2 * r * h * bc1.value(ti)
            b[-1] += r * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def crank_nicolson(bc1, bc2, M, N, t_end, u0=initial, log=True):
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
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
        # Neuman-Neuman
        diag = np.repeat(1 + r, M)
        offdiag_upper = np.repeat(-r / 2, M - 1)
        offdiag_lower = np.repeat(-r / 2, M - 1)
        diag[0] = 1 + r / 2
        offdiag_upper[0] = -r / 2
        diag[-1] = 1 + r / 2
        offdiag_lower[-1] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = np.zeros(M)
            b[1:-1] = (r / 2) * U[:-2] + (1 - r) * U[1:-1] + (r / 2) * U[2:]
            b[0] = U[0] + (r / 2) * (U[1] - U[0]) - 2 * r * h * bc1.value(ti)
            b[-1] = U[-1] + (r / 2) * (U[-2] - U[-1]) + 2 * r * h * bc2.value(ti)
            U = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # DIRCHLET-DIRCHLET
        m = M - 2
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-2] + (1 - r) * U[1:-1] + (r / 2) * U[2:]
            b[0] += (r / 2) * bc1.value(ti)
            b[-1] += (r / 2) * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # NEUMANN-DIRCHLET
        print("WARNING: Case Neumann-Dirchlet not yet tested, werid stuff may happen!")
        m = M - 1
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        diag[0] = 1 + r / 2
        offdiag_upper[0] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-1] + (1 - r) * U[:-1] + (r / 2) * U[1:]
            b[0] += (r / 2) * (U[1] - U[0]) - 2 * r * h * bc1.value(ti)
            b[-1] += (r / 2) * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
        if log:
            return x, t, U, solution_matrix
    elif (
        bc1.type == BoundaryCondition.DIRCHLET and bc2.type == BoundaryCondition.NEUMANN
    ):
        # DIRCHLET-NEUMANN
        print("WARNING: Case Dirchlet-Neumann not yet tested, werid stuff may happen!")
        m = M - 1
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        diag[-1] = 1 + r / 2
        offdiag_lower[-1] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-1] + (1 - r) * U[1:] + (r / 2) * U[1:]
            b[0] += (r / 2) * bc1.value(ti)
            b[-1] += (r / 2) * (U[-2] - U[-1]) + 2 * r * h * bc2.value(ti)
            U[1:] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def test_method(method, M, N, t_end):
    """
    Do a testrun and plot results for a numerical solver of the heat equation

    Parameters:
        method : the function name of the method to be tested e.g. forward_euler
        M : number of spacial grid points
        N : number of time grid points
        t_end : end time
    """

    # bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, initial(0))
    # bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, initial(1))
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    x, t, U_final, solutions = method(bc1, bc2, M, N, t_end)

    num_samples = 5
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(x, solutions[j], ".", label=f"t={ti}")
    plt.title(method.__name__)
    plt.legend()
    plt.show()


def discrete_l2_norm(V):
    """ discrete l2 norm """
    return np.linalg.norm(V) / np.sqrt(len(V))


def l2_discrete_relative_error(U_ref, U):
    """ Compute and return the l2 discrete relative error """

    return discrete_l2_norm(U_ref - U) / discrete_l2_norm(U_ref)


def L2_continous_norm(v, x_min=0, x_max=1):
    """ Compute and return the L2 continous norm """
    return np.sqrt(quad(lambda x: v(x) ** 2, x_min, x_max)[0])


def L2_continous_relative_error(U_ref, U):
    """ Compute and return the L2 continous relative error """
    return L2_continous_norm(lambda x: U_ref(x) - U(x)) / L2_continous_norm(U_ref)


def piecewise_constant_continuation(xr, ur):
    """
    make a piecewise constant function of spacial coordinate x from a reference solution u

    Parameters:
        xr : x grid for the reference solution
        ur : Array, the reference solution
    Returns:
        numpy.piecewise function, piecewise constant funciton of x
    """

    return lambda x: np.piecewise(
        x,
        [xr[i] <= x < xr[j] for (i, j) in zip(range(len(ur) - 1), range(1, len(ur)))],
        ur,
    )


def continous_continuation(xr, ur):
    """ Cont. continuation using interpolation """

    return lambda x: np.interp(x, xr, ur)


def discrete_convergence_plot(method, M_ref, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """

    # Neuman BC's
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 1)

    # Reference solution (in place of analytical)
    ref_x, _, ref_sol = method(
        bc1, bc2, M_ref, N, t_end, log=False
    )  # reference sol in array form
    u = np.vectorize(
        piecewise_constant_continuation(ref_x, ref_sol)
    )  # reference sol, piece wise constant callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(45, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, M) in enumerate(M_array):
        x, _, U = method(bc1, bc2, M, N, t_end, log=False)  # solution with current M
        U_ref = u(x)  # Discretized reference solution
        error_array[i] = l2_discrete_relative_error(U_ref, U)  # dicrete relative error
    plt.xlabel("M")
    plt.ylabel("rel. error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(M_array, error_array)
    plt.show()


def continous_convergence_plot(method, M_ref, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """

    # Neuman BC's
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 1)

    # Reference solution (in place of analytical)
    ref_x, _, ref_sol = method(
        bc1, bc2, M_ref, N, t_end, log=False
    )  # reference sol in array form
    U_ref = continous_continuation(
        ref_x, ref_sol
    )  # reference sol, callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(45, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, M) in enumerate(M_array):
        x, _, U_array = method(
            bc1, bc2, M, N, t_end, log=False
        )  # solution with current M
        U = continous_continuation(x, U_array)
        error_array[i] = L2_continous_relative_error(
            U_ref, U
        )  # continous relative error
    plt.xlabel("M")
    plt.ylabel("rel. error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(M_array, error_array)
    plt.show()


def test():
    """ Solves and plots solution at different sample times """
    ## Test forward Euler ##
    test_method(forward_euler, 100, 10000, 0.2)
    ## Test backward Euler
    test_method(backward_euler, 100, 100, 0.2)
    ## Test Crank-Nicolson
    test_method(crank_nicolson, 100, 100, 0.2)


def make_discrete_convergence_plots():
    """ 2a) """
    #discrete_convergence_plot(forward_euler, 1000, 200, 100, 0.2)
    discrete_convergence_plot(backward_euler, 10000, 200, 100, 0.2)
    discrete_convergence_plot(crank_nicolson, 1000, 200, 100, 0.2)


def make_continous_convergence_plots():
    """ 2a) """
    #continous_convergence_plot(forward_euler, 1000, 200, 100, 0.2)
    continous_convergence_plot(backward_euler, 1000, 200, 100, 0.2)
    continous_convergence_plot(crank_nicolson, 1000, 200, 100, 0.2)


def task2b():
    # Set up problem
    bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
    bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
    def u0(x, x_min=0, x_max=1):
        mid = (x_max - x_min)/2
        if x_min <= x and x <= mid:
            return x
        elif mid < x and x <= x_max:
            return (x_max - x)
        else:
            return 0
    u0 = np.vectorize(u0)
    """
    x = np.linspace(0,1, 50)
    plt.plot(x, u0(x))
    plt.show()
    """
    
    # Test solvers for the new equation/conditions
    x, t, U, solutions = forward_euler(bc1, bc2, 100, 10000, 0.1, u0=u0)
    plt.title("FE")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()

    x, t, U, solutions = backward_euler(bc1, bc2, 100, 10000, 0.1, u0=u0)
    plt.title("BE")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()

    x, t, U, solutions = crank_nicolson(bc1, bc2, 100, 100, 0.1, u0=u0)
    plt.title("CN")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #test()
    #make_discrete_convergence_plots()
    #make_continous_convergence_plots()
    task2b()
