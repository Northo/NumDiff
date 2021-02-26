#!/usr/bin/python3

""" Task 2 -- Heat equation """
import numpy as np
from matplotlib import pyplot as plt

from utils import BoundaryCondition, Grid
from routines import forward_euler, backward_euler, crank_nicolson


def test_method(method, M, N, t_end):
    """
    Do a testrun and plot results for a numerical solver of the heat equation.
    For "veryfying" that a method works.
    Uses initial and boundary conditions from task 2a)

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

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    x, t, U_final, solutions = method(bc1, bc2, u0, M, N, t_end)

    num_samples = 5
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(x, solutions[j], ".", label=f"t={ti}")
    plt.title(method.__name__)
    plt.legend()
    plt.show()


def discrete_convergence_plot(method, bc1, bc2, u0, M_ref, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """

    # Reference solution (in place of analytical)
    ref_x, _, ref_sol = method(
        bc1, bc2, u0, M_ref, N, t_end, log=False
    )  # reference sol in array form
    u = np.vectorize(
        piecewise_constant_continuation(ref_x, ref_sol)
    )  # reference sol, piece wise constant callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(10, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, M) in enumerate(M_array):
        x, _, U = method(bc1, bc2, u0, M, N, t_end, log=False)  # solution with current M
        U_ref = u(x)  # Discretized reference solution
        error_array[i] = l2_discrete_relative_error(U_ref, U)  # dicrete relative error
    plt.xlabel("M")
    plt.ylabel("rel. error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(M_array, error_array)
    plt.show()


def continous_convergence_plot(method, bc1, bc2, u0, M_ref, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """

    # Neuman BC's
    #bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    #bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    # Reference solution (in place of analytical)
    ref_x, _, ref_sol = method(
        bc1, bc2, u0, M_ref, N, t_end, log=False
    )  # reference sol in array form
    U_ref = continous_continuation(
        ref_x, ref_sol
    )  # reference sol, callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(10, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, M) in enumerate(M_array):
        x, _, U_array = method(
            bc1, bc2, u0, M, N, t_end, log=False
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

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    discrete_convergence_plot(backward_euler, bc1, bc2, u0, 1000, 200, 100, 0.2)
    discrete_convergence_plot(crank_nicolson, bc1, bc2, u0, 1000, 200, 100, 0.2)


def make_continous_convergence_plots():
    """ 2a) """

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    continous_convergence_plot(backward_euler, bc1, bc2, u0, 1000, 200, 100, 0.2)
    continous_convergence_plot(crank_nicolson, bc1, bc2, u0, 1000, 200, 100, 0.2)


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
    
    # Test solvers for the new equation/conditions
    x, t, U, solutions = forward_euler(bc1, bc2, u0, 100, 10000, 0.1)
    plt.title("FE")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()

    x, t, U, solutions = backward_euler(bc1, bc2, u0, 100, 10000, 0.1)
    plt.title("BE")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()

    x, t, U, solutions = crank_nicolson(bc1, bc2, u0, 100, 100, 0.1)
    plt.title("CN")
    plt.plot(x, solutions[0], label="initial")
    plt.plot(x, U, label="final")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #test()
    #make_discrete_convergence_plots()
    #make_continous_convergence_plots()
    #task2b()
    test_method(forward_euler, 100, 10000, 0.2)
    test_method(backward_euler, 100, 100, 0.2)
    test_method(crank_nicolson, 100, 100, 0.2)
