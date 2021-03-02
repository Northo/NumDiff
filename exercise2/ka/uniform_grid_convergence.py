#!/usr/bin/python3

""" Task 2 -- Heat equation """
import numpy as np
from matplotlib import pyplot as plt

from utils import l2_discrete_relative_error, L2_continous_relative_error, piecewise_constant_continuation, continous_continuation
from routines import BoundaryCondition, forward_euler, backward_euler, crank_nicolson, test_method, Grid


def discrete_convergence_plot(method, ref_grid, bc1, bc2, u0, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """
    # Reference solution (in place of analytical)
    _, ref_sol = method(
        ref_grid, bc1, bc2, u0, N, t_end, log=False
    )  # reference sol in array form
    u = np.vectorize(
        piecewise_constant_continuation(ref_grid.x, ref_sol)
    )  # reference sol, piece wise constant callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(45, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, Mi) in enumerate(M_array):
        grid_Mi = Grid(Grid.UNIFORM, np.linspace(0, 1, Mi))
        _, U = method(grid_Mi, bc1, bc2, u0, N, t_end, log=False)  # solution with current M
        U_ref = u(grid_Mi.x) # Discretized reference solution
        error_array[i] = l2_discrete_relative_error(U_ref, U)  # dicrete relative error
    plt.xlabel("M")
    plt.ylabel("rel. error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(M_array, error_array)
    plt.show()


def continous_convergence_plot(method, ref_grid, bc1, bc2, u0, M_max, N, t_end):
    """ Make convergence (plot relative error asf. of M) """
    # Reference solution (in place of analytical)
    _, ref_sol = method(
        ref_grid, bc1, bc2, u0, N, t_end, log=False
    )  # reference sol in array form
    U_ref = continous_continuation(
        ref_grid.x, ref_sol
    )  # reference sol, callable function

    # Different M values (for parameter sweep)
    M_array = np.arange(45, M_max, 10)
    error_array = np.zeros(len(M_array))  # for storing relative errors

    for (i, Mi) in enumerate(M_array):
        grid_Mi = Grid(Grid.UNIFORM, np.linspace(0, 1, Mi))
        _, U_array = method(
            grid_Mi, bc1, bc2, u0, N, t_end, log=False
        )  # solution with current M
        U = continous_continuation(grid_Mi.x, U_array)
        error_array[i] = L2_continous_relative_error(
            U_ref, U
        )  # continous relative error
    plt.xlabel("M")
    plt.ylabel("rel. error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(M_array, error_array)
    plt.show()


def make_discrete_convergence_plots():
    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0,1,M_ref))
    discrete_convergence_plot(backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.1)
    discrete_convergence_plot(crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.1)


def make_continous_convergence_plots():
    """ 2a) """

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0,1,M_ref))
    continous_convergence_plot(backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.2)
    continous_convergence_plot(crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.2)


if __name__ == "__main__":
    make_discrete_convergence_plots()
    make_continous_convergence_plots()
