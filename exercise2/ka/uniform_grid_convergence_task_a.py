#!/usr/bin/python3

""" Task 2 -- Heat equation """
import numpy as np
from matplotlib import pyplot as plt

from utils import (
    l2_discrete_relative_error,
    L2_continous_relative_error,
    piecewise_constant_continuation,
    continous_continuation,
    discrete_convergence_plot,
    continous_convergence_plot,
)
from routines import (
    BoundaryCondition,
    forward_euler,
    backward_euler,
    crank_nicolson,
    test_method,
    Grid,
)


def make_discrete_convergence_plots():
    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    discrete_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.1, plot=True
    )
    discrete_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.1, plot=True
    )


def make_continous_convergence_plots():
    """ 2a) """

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    continous_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.2, plot=True
    )
    continous_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.2, plot=True
    )


if __name__ == "__main__":
    make_discrete_convergence_plots()
    make_continous_convergence_plots()
