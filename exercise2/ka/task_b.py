#!/usr/bin/python3
from heateq import *

PLOT_SAMPLES = True
OUT_DIR = "../../report/exercise2/data_ka/"


##################################
### Defining equation - IC/BCs ###
##################################


def u0(x):
    """ Initial condition """
    # return 10*np.sin(x*np.pi)
    return np.sin(x * np.pi)


# Boundary conditions (Dirchlet)
bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)


def analytical(x, t):
    """ Analytical solution of manufactured dirchlet problem """
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


##########################
### Numerical solution ###
##########################


# Setup grid and stuff
unigrid = Grid(Grid.UNIFORM, np.linspace(0, 1, 100)) # spatial uniform grid
N = 100 # Number of timesteps
t_end = 0.5 # Final/end time

# Solve equation numerically
t, U_final, sols = crank_nicolson(unigrid, bc1, bc2, u0, N, t_end)


if PLOT_SAMPLES:
    n_samples = 5
    for i in range(n_samples):
        j = i * N // n_samples
        ti = t[j]
        plt.plot(unigrid.x, sols[j], ".", label=f"num, t={ti}")
        plt.plot(unigrid.x, analytical(unigrid.x, ti), label=f"analyt, t={ti}")
    plt.show()


###############################################
### Compare numerical and analytical -- UMR ###
###############################################


def make_UMR_convergence_plots(u0, bc1, bc2):
    N = 100
    outpath=f"{OUT_DIR}2b_UMR_BE_discrete_err.dat"
    discrete_convergence_plot(
        analytical, backward_euler, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_UMR_CN_discrete_err.dat"
    discrete_convergence_plot(
        analytical, crank_nicolson, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_UMR_BE_continous_err.dat"
    continous_convergence_plot(
        analytical, backward_euler, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_UMR_CN_continous_err.dat"
    continous_convergence_plot(
        analytical, crank_nicolson, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )


make_UMR_convergence_plots(u0, bc1, bc2)


###########
### AMR ###
###########


def error_func(a, c, b, u0=u0):
    """ Estimate of meassure of error of some sort or something """
    return np.abs(u0(c) - (u0(a) + u0(b)) / 2)


# Test solving on non uniform (adaptive) grid
x_adapt = partition_interval(0, 1, error_func, 0.001)

adapt_grid = Grid(Grid.NON_UNIFORM, x_adapt)
t, U_final, sols = crank_nicolson(adapt_grid, bc1, bc2, u0, N, t_end)

plt.plot(x_adapt, np.zeros(len(x_adapt)), ".")
plt.plot(x_adapt, U_final)
plt.title(f"Adaptive grid, # points: {len(x_adapt)}")
plt.show()


def make_AMR_convergence_plots(u0, bc1, bc2):
    N = 100
    outpath=f"{OUT_DIR}2b_AMR_BE_discrete_err.dat"
    AMR_discrete_convergence_plot(
        error_func, analytical, backward_euler, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_AMR_CN_discrete_err.dat"
    AMR_discrete_convergence_plot(
        error_func, analytical, crank_nicolson, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_AMR_BE_continous_err.dat"
    AMR_continous_convergence_plot(
        error_func, analytical, backward_euler, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )
    outpath=f"{OUT_DIR}2b_AMR_CN_continous_err.dat"
    AMR_continous_convergence_plot(
        error_func, analytical, crank_nicolson, bc1, bc2, u0, N, 1, plot=True, outpath=outpath
    )


make_AMR_convergence_plots(u0, bc1, bc2)

### Animation ###
# animation = animate_time_development(unigrid.x, sols)
# plt.show()
