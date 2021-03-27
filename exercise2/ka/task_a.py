#!/usr/bin/python3
""" Task 2 -- Heat equation """
from heateq import *

OUT_DIR = "../../report/exercise2/data_ka/"


def make_discrete_convergence_plots(u0, bc1, bc2):
    M_ref = 1000
    N = 100
    t_end = 1
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    outpath = f"{OUT_DIR}2a_BE_discrete_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    discrete_convergence_plot_M_ref(
        backward_euler, ref_grid, bc1, bc2, u0, N, t_end, plot=True, outpath=outpath
    )
    outpath = f"{OUT_DIR}2a_CN_discrete_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    discrete_convergence_plot_M_ref(
        crank_nicolson, ref_grid, bc1, bc2, u0, N, t_end, plot=True, outpath=outpath
    )


def make_continous_convergence_plots(u0, bc1, bc2):
    M_ref = 1000
    N = 100
    t_end = 1
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    outpath = f"{OUT_DIR}2a_BE_continous_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    continous_convergence_plot_M_ref(
        backward_euler, ref_grid, bc1, bc2, u0, N, t_end, plot=True, outpath=outpath
    )
    outpath = f"{OUT_DIR}2a_CN_continous_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    continous_convergence_plot_M_ref(
        crank_nicolson, ref_grid, bc1, bc2, u0, N, t_end, plot=True, outpath=outpath
    )


if __name__ == "__main__":

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    # Commented out to avoid accidental data overwrite
    # make_discrete_convergence_plots(u0, bc1, bc2)
    # make_continous_convergence_plots(u0, bc1, bc2)

    grid = Grid(Grid.UNIFORM, np.linspace(0, 1, 100))
    t, U_final, sols = crank_nicolson(grid, bc1, bc2, u0, 100, 0.5)
    #t, U_final, sols = backward_euler(grid, bc1, bc2, u0, 100, 0.5)

    #u_table = [sols[i] for i in t]
    U_table = np.resize(sols, sols.size)
    x_table = np.tile(grid.x, len(t))
    t_table = np.repeat(t, len(grid.x))
    table = np.column_stack((x_table, t_table, U_table))
    outpath = f"{OUT_DIR}2a_surface.dat"
    np.savetxt(outpath, table, header="x t U", comments="")


    # Animation
    animation = animate_time_development(grid.x, sols)
    plt.show()
