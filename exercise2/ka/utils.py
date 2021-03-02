import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from matplotlib import animation
from collections.abc import Callable  # Spooky stuff I don't know what is

from routines import Grid


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


def discrete_convergence_plot(
    method, ref_grid, bc1, bc2, u0, M_max, N, t_end, plot=False
):
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
        _, U = method(
            grid_Mi, bc1, bc2, u0, N, t_end, log=False
        )  # solution with current M
        U_ref = u(grid_Mi.x)  # Discretized reference solution
        error_array[i] = l2_discrete_relative_error(U_ref, U)  # dicrete relative error
    if plot:
        plt.xlabel("M")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(M_array, error_array)
        plt.show()
    return error_array, M_array


def continous_convergence_plot(
    method, ref_grid, bc1, bc2, u0, M_max, N, t_end, plot=False
):
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
    if plot:
        plt.xlabel("M")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(M_array, error_array)
        plt.show()
    return error_array, M_array


# From ../jonas.py
def animate(i, x, U, curve):
    curve.set_data(x, U[i])
    return curve


# From ../jonas.py
def animate_time_development(x, U):
    """
    Animates values of U developing in N time steps.
    Variables:
        x:  np.array of x values in which function is evaluated.
        U:  np.array with dimensions (N, M) holding
            M function values in N time steps.
    """
    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(0, np.max(U) * 1.1))
    (curve,) = ax.plot(x, U[0])
    anim = animation.FuncAnimation(fig, animate, fargs=(x, U, curve))
    return anim


# From T-vice
def _split_interval(a, b, error_function, tol):
    """Helper function used by partition_interval"""
    c = (a + b) / 2  # Bisection
    if error_function(a, c, b) <= tol:
        partition = [c]
    else:
        partition = [
            *_split_interval(a, c, error_function, tol),
            c,
            *_split_interval(c, b, error_function, tol),
        ]
    return partition


# From T-vice
def partition_interval(
    a, b, error_function: Callable[[float, float, float], float], tol
):
    """Partition an interval adaptively.
    Makes error_function less than tol for all sub intervals.
    Arguments:
        a,b : float The start and stop of the interval.
        errror_function : func(a, c, b) -> err, error estimation for the interval [a, b].
        tol : float The tolerance for the error on an interval.
    Returns:
        x : ndarray The partitioned interval."""
    x = _split_interval(a, b, error_function, tol)
    return np.array([a, *x, b])
