#!/usr/bin/python3
from heateq import *
from functools import partial

PLOT_SAMPLES = True
OUT_DIR = "../../report/exercise2/data_ka/"


def make_spatial_convergence_plots(u0, bc1, bc2, error_type, analyt, N=10000):
    t_end = 1
    outpath = f"{OUT_DIR}2b_BE_spatialref_{error_type}_err_N{N}_tend{t_end}.dat"
    print(outpath)
    spatial_refinement(
        backward_euler,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        N,
        t_end,
        plot=True,
        outpath=outpath,
    )

    outpath = f"{OUT_DIR}2b_CN_spatialref_{error_type}_err_N{N}_tend{t_end}.dat"
    spatial_refinement(
        crank_nicolson,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        N,
        t_end,
        plot=True,
        outpath=outpath,
    )


def make_temporal_convergence_plots(u0, bc1, bc2, error_type, analyt, M=10000):
    t_end = 1
    outpath = f"{OUT_DIR}2b_BE_timeref_{error_type}_err_M{M}_tend{t_end}.dat"
    temporal_refinement(
        backward_euler,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        M,
        t_end,
        plot=True,
        outpath=outpath,
    )

    outpath = f"{OUT_DIR}2b_CN_timeref_{error_type}_err_M{M}_tend{t_end}.dat"
    temporal_refinement(
        crank_nicolson,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        M,
        t_end,
        plot=True,
        outpath=outpath,
    )


def make_kch_convergence_plots(u0, bc1, bc2, error_type, analyt, c=1):
    t_end = 1
    outpath = f"{OUT_DIR}2b_BE_kchref_{error_type}_err_c{c}_tend{t_end}.dat"
    kch_refinement(
        backward_euler,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        c,
        t_end,
        plot=True,
        outpath=outpath,
    )

    outpath = f"{OUT_DIR}2b_CN_kchref_{error_type}_err_c{c}_tend{t_end}.dat"
    kch_refinement(
        crank_nicolson,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        c,
        t_end,
        plot=True,
        outpath=outpath,
    )


def make_r_convergence_plots(u0, bc1, bc2, error_type, analyt, r=1):
    t_end = 1
    outpath = f"{OUT_DIR}2b_BE_rref_{error_type}_err_r{r}_tend{t_end}.dat"
    r_refinement(
        backward_euler,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        r,
        t_end,
        plot=True,
        outpath=outpath,
    )

    outpath = f"{OUT_DIR}2b_CN_rref_{error_type}_err_r{r}_tend{t_end}.dat"
    r_refinement(
        crank_nicolson,
        analyt,
        error_type,
        bc1,
        bc2,
        u0,
        r,
        t_end,
        plot=True,
        outpath=outpath,
    )


if __name__ == "__main__":

    def u0(x):
        """ Initial condition """
        # return 10*np.sin(x*np.pi)
        return np.sin(x * np.pi)

    # Boundary conditions (Dirchlet)
    bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
    bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)

    def analytical(x, t):
        """ Analytical solution of manufactured dirchlet problem """
        return np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)

#    make_spatial_convergence_plots(u0, bc1, bc2, "discrete", analytical, N=10000)
#    make_spatial_convergence_plots(u0, bc1, bc2, "continous", analytical, N=10000)
#    make_temporal_convergence_plots(u0, bc1, bc2, "discrete", analytical, M=10000)
#    make_temporal_convergence_plots(u0, bc1, bc2, "continous", analytical, M=10000)
    make_kch_convergence_plots(u0, bc1, bc2, "discrete", analytical, c=2)
    make_kch_convergence_plots(u0, bc1, bc2, "continous", analytical, c=2)
    make_r_convergence_plots(u0, bc1, bc2, "discrete", analytical, r=2)
    make_r_convergence_plots(u0, bc1, bc2, "continous", analytical, r=2)

    #x = np.linspace(0, 1, 50)
    #t, U_final, sols = theta_heat(bc1, bc2, u0, x, 50, 0.25, method="cn")
    #outpath = f"{OUT_DIR}2b_surface.dat"
    #save_solution_surface_plot_data(x, t, sols, outpath)

    # Animation
    #animation = animate_time_development(x, sols)
    #plt.show()
