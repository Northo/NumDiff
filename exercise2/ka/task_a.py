#!/usr/bin/python3
""" Task 2 -- Heat equation """
from heateq import *

# OUT_DIR = "../../report/exercise2/data_ka/"
OUT_DIR = "./data/"


def make_reference_convergence_plots(u0, bc1, bc2, error_type, M_ref):
    N = 100
    t_end = 1
    outpath = f"{OUT_DIR}2a_BE_{error_type}_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    reference_spatial_refinement(
        backward_euler,
        M_ref,
        error_type,
        bc1,
        bc2,
        u0,
        N,
        t_end,
        plot=True,
        outpath=outpath,
    )

    outpath = f"{OUT_DIR}2a_CN_{error_type}_err_N{N}_Mref{M_ref}_tend{t_end}.dat"
    reference_spatial_refinement(
        crank_nicolson,
        M_ref,
        error_type,
        bc1,
        bc2,
        u0,
        N,
        t_end,
        plot=True,
        outpath=outpath,
    )


if __name__ == "__main__":

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    make_reference_convergence_plots(u0, bc1, bc2, "discrete", 1000)
    make_reference_convergence_plots(u0, bc1, bc2, "continous", 1000)

    x = np.linspace(0, 1, 100)
    t, U_final, sols = theta_heat(bc1, bc2, u0, x, 100, 0.5, method="cn")

    outpath = f"{OUT_DIR}2a_surface.dat"
    save_solution_surface_plot_data(x, t, sols, outpath)

    #    U_table = np.resize(sols, sols.size)
    #    x_table = np.tile(x, len(t))
    #    t_table = np.repeat(t, len(x))
    #    table = np.column_stack((x_table, t_table, U_table))
    #    outpath = f"{OUT_DIR}2a_surface.dat"
    #    np.savetxt(outpath, table, header="x t U", comments="")

    # Animation
    animation = animate_time_development(x, sols)
    plt.show()
