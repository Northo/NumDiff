#!/usr/bin/python3
""" Task 2 -- Heat equation """
from heateq import *


def make_discrete_convergence_plots(u0, bc1, bc2):
    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    discrete_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.5, plot=True
    )
    discrete_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.5, plot=True
    )


def make_continous_convergence_plots(u0, bc1, bc2):
    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    continous_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 0.5, plot=True
    )
    continous_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 0.5, plot=True
    )


if __name__ == "__main__":

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    make_discrete_convergence_plots(u0, bc1, bc2)
    make_continous_convergence_plots(u0, bc1, bc2)

    grid = Grid(Grid.UNIFORM, np.linspace(0, 1, 100))
    t, U_final, sols = crank_nicolson(grid, bc1, bc2, u0, 100, 0.5)

    # Animation
    animation = animate_time_development(grid.x, sols)
    plt.show()
