import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from matplotlib import animation
from collections.abc import Callable  # Spooky stuff I don't know what is

from functools import partial


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


def reference_spatial_refinement(
    n_solver, MN_ref, error_type, bc1, bc2, u0, N, t_end, plot=False, outpath=""
):
    M_ref = MN_ref
    N_ref = MN_ref
    x_ref = np.linspace(0, 1, M_ref)
    _, sol_ref = n_solver(bc1, bc2, u0, x_ref, N_ref, t_end, log=False)
    if error_type == "discrete":
        sol_ref_pwc = np.vectorize(piecewise_constant_continuation(x_ref, sol_ref))
    elif error_type == "continous":
        U_ref = continous_continuation(x_ref, sol_ref)
    M_array = [8, 16, 32, 64, 128, 256, 512, 1024]
    error_array = np.zeros(len(M_array))  # for storing relative errors
    for (i, Mi) in enumerate(M_array):
        xi = np.linspace(0, 1, Mi)
        _, U = n_solver(bc1, bc2, u0, xi, N, t_end, log=False)
        if error_type == "discrete":
            U_ref = sol_ref_pwc(xi)
            error_array[i] = l2_discrete_relative_error(U_ref, U)
        elif error_type == "continous":
            U = continous_continuation(xi, U)
            error_array[i] = L2_continous_relative_error(U_ref, U)
    if outpath != "":
        table = np.column_stack((M_array, error_array))
        np.savetxt(outpath, table, header="M err", comments="")
        plt.title(outpath)
    if plot:
        plt.title(n_solver.__name__ + "spatialref" + error_type)
        plt.xlabel("M")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(M_array, error_array)
        plt.plot(M_array, error_array, "x")
        plt.show()


def spatial_refinement(
    n_solver, analyt, error_type, bc1, bc2, u0, N, t_end, plot=False, outpath=""
):
    u_analytical = lambda x: analyt(x, t_end)
    M_array = [8, 16, 32, 64, 128, 256, 512, 1024]
    error_array = np.zeros(len(M_array))  # for storing relative errors
    for (i, Mi) in enumerate(M_array):
        xi = np.linspace(0, 1, Mi)
        _, U = n_solver(bc1, bc2, u0, xi, N, t_end, log=False)
        if error_type == "discrete":
            U_ref = u_analytical(xi)
            error_array[i] = l2_discrete_relative_error(U_ref, U)
        elif error_type == "continous":
            U = continous_continuation(xi, U)
            error_array[i] = L2_continous_relative_error(u_analytical, U)
    if outpath != "":
        table = np.column_stack((M_array, error_array))
        np.savetxt(outpath, table, header="M err", comments="")
        plt.title(outpath)
    if plot:
        plt.title(n_solver.__name__ + "spatialref" + error_type)
        plt.xlabel("M")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(M_array, error_array)
        plt.plot(M_array, error_array, "x")
        plt.show()


def temporal_refinement(
    n_solver, analyt, error_type, bc1, bc2, u0, M, t_end, plot=False, outpath=""
):
    u_analytical = lambda x: analyt(x, t_end)
    x = np.linspace(0, 1, M)
    N_array = [8, 16, 32, 64, 128, 256, 512, 1024]
    error_array = np.zeros(len(N_array))  # for storing relative errors
    for (i, Ni) in enumerate(N_array):
        _, U = n_solver(bc1, bc2, u0, x, Ni, t_end, log=False)
        if error_type == "discrete":
            U_ref = u_analytical(x)
            error_array[i] = l2_discrete_relative_error(U_ref, U)
        elif error_type == "continous":
            U = continous_continuation(x, U)
            error_array[i] = L2_continous_relative_error(u_analytical, U)
    if outpath != "":
        table = np.column_stack((N_array, error_array))
        np.savetxt(outpath, table, header="N err", comments="")
        plt.title(outpath)
    if plot:
        plt.title(n_solver.__name__ + "timeref" + error_type)
        plt.xlabel("M")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(N_array, error_array)
        plt.plot(N_array, error_array, "x")
        plt.show()


# Optimal refinement for Crank-Nicolson
def kch_refinement(
    n_solver, analyt, error_type, bc1, bc2, u0, c, t_end, plot=False, outpath=""
):
    u_analytical = lambda x: analyt(x, t_end)
    M_array = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    h_array = 1 / (M_array - 1)
    k_array = c * h_array
    N_array = 1 / k_array + 1
    Ndof_array = M_array * N_array
    error_array = np.zeros(len(M_array))  # for storing relative errors
    for (i, (Mi, Ni)) in enumerate(zip(M_array, N_array)):
        Ni = int(Ni)
        xi = np.linspace(0, 1, Mi)
        _, U = n_solver(bc1, bc2, u0, xi, Ni, t_end, log=False)
        if error_type == "discrete":
            U_ref = u_analytical(xi)
            error_array[i] = l2_discrete_relative_error(U_ref, U)
        elif error_type == "continous":
            U = continous_continuation(xi, U)
            error_array[i] = L2_continous_relative_error(u_analytical, U)
    if outpath != "":
        table = np.column_stack((Ndof_array, M_array, N_array, error_array))
        np.savetxt(outpath, table, header="Ndof M N err", comments="")
        plt.title(outpath)
    if plot:
        plt.title(n_solver.__name__ + "kchref" + error_type)
        plt.xlabel("M*N(Ndof)")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(Ndof_array, error_array)
        plt.plot(Ndof_array, error_array, "x")
        plt.show()


# Optimal refinement for Backward-Euler
def r_refinement(
    n_solver, analyt, error_type, bc1, bc2, u0, r, t_end, plot=False, outpath=""
):
    u_analytical = lambda x: analyt(x, t_end)
    M_array = np.array([8, 16, 32, 64, 128, 256])
    h_array = 1 / (M_array - 1)
    k_array = r * (h_array **2)
    N_array = 1 / k_array + 1
    Ndof_array = M_array * N_array
    error_array = np.zeros(len(M_array))  # for storing relative errors
    for (i, (Mi, Ni)) in enumerate(zip(M_array, N_array)):
        Ni = int(Ni)
        xi = np.linspace(0, 1, Mi)
        _, U = n_solver(bc1, bc2, u0, xi, Ni, t_end, log=False)
        if error_type == "discrete":
            U_ref = u_analytical(xi)
            error_array[i] = l2_discrete_relative_error(U_ref, U)
        elif error_type == "continous":
            U = continous_continuation(xi, U)
            error_array[i] = L2_continous_relative_error(u_analytical, U)
    if outpath != "":
        table = np.column_stack((Ndof_array, M_array, N_array, error_array))
        np.savetxt(outpath, table, header="Ndof M N err", comments="")
        plt.title(outpath)
    if plot:
        plt.title(n_solver.__name__ + "rref" + error_type)
        plt.xlabel("M*N(Ndof)")
        plt.ylabel("rel. error")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(Ndof_array, error_array)
        plt.plot(Ndof_array, error_array, "x")
        plt.show()


def save_solution_surface_plot_data(x, t, sols, outpath):
    U_table = np.resize(sols, sols.size)
    x_table = np.tile(x, len(t))
    t_table = np.repeat(t, len(x))
    table = np.column_stack((x_table, t_table, U_table))
    np.savetxt(outpath, table, header="x t U", comments="")


def save_solution_sample_plot(x, t, sols, outpath):
    num_samples = 8
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(x, solutions[j], ".", label=f"t={ti}")
    plt.title(method)
    plt.legend()
    plt.show()


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


# Stuff below is for possible extenison to non-uniform grids and AMR
# From T-vice
# def _split_interval(a, b, error_function, tol):
#    """Helper function used by partition_interval"""
#    c = (a + b) / 2  # Bisection
#    if error_function(a, c, b) <= tol:
#        partition = [c]
#    else:
#        partition = [
#            *_split_interval(a, c, error_function, tol),
#            c,
#            *_split_interval(c, b, error_function, tol),
#        ]
#    return partition
#
#
## From T-vice
# def partition_interval(
#        a, b, tol, error_function: Callable[[float, float, float], float]
# ):
#    """Partition an interval adaptively.
#    Makes error_function less than tol for all sub intervals.
#    Arguments:
#        a,b : float The start and stop of the interval.
#        errror_function : func(a, c, b) -> err, error estimation for the interval [a, b].
#        tol : float The tolerance for the error on an interval.
#    Returns:
#        x : ndarray The partitioned interval."""
#    x = _split_interval(a, b, error_function, tol)
#    return np.array([a, *x, b])
#
#
# def AMR_discrete_convergence_plot(
#    error_function, analy, method, bc1, bc2, u0, N, t_end, plot=False, outpath=""
# ):
#    M_array = [8, 16, 32, 64, 128, 256, 512, 1024]  # Ms from UMR
#    error_array = np.empty(len(M_array))
#    tols = np.geomspace(0.0000046, 0.1, len(M_array))
#    u_analy = lambda x: analy(x, t_end)  # analytical sol at t_end
#    actual_M = []
#    for (i, tol) in enumerate(tols):
#        x = partition_interval(0, 1, error_function, tol)
#        actual_M.append(len(x))
#        grid_Mi = Grid(Grid.NON_UNIFORM, x)
#        _, U = theta_heat(
#            grid_Mi, bc1, bc2, u0, N, t_end, log=False, method=method
#        )  # solution with current M
#        U_ref = u_analy(grid_Mi.x)  # Discretized reference solution
#        error_array[i] = l2_discrete_relative_error(U_ref, U)  # dicrete relative error
#    if outpath != "":
#        table = np.column_stack((actual_M, error_array))
#        np.savetxt(outpath, table, header="M err", comments="")
#        plt.title(outpath)
#    if plot:
#        plt.xlabel("M")
#        plt.ylabel("rel. error")
#        plt.xscale("log")
#        plt.yscale("log")
#        plt.plot(actual_M, error_array)
#        plt.plot(actual_M, error_array, "x")
#        plt.show()
#    return error_array, M_array
#
#
# def AMR_continous_convergence_plot(
#    error_function, analy, method, bc1, bc2, u0, N, t_end, plot=False, outpath=""
# ):
#    M_array = [8, 16, 32, 64, 128, 256, 512, 1024]  # Ms from UMR
#    error_array = np.empty(len(M_array))
#    tols = np.geomspace(0.0000046, 0.1, len(M_array))
#    U_ref = lambda x: analy(x, t_end)  # analytical sol at t_end
#    actual_M = []
#    for (i, tol) in enumerate(tols):
#        x = partition_interval(0, 1, error_function, tol)
#        actual_M.append(len(x))
#        grid_Mi = Grid(Grid.NON_UNIFORM, x)
#        _, U_array = theta_heat(
#            grid_Mi, bc1, bc2, u0, N, t_end, log=False, method=method
#        )  # solution with current M
#        U = continous_continuation(grid_Mi.x, U_array)
#        error_array[i] = L2_continous_relative_error(U_ref, U)  # dicrete relative error
#    if outpath != "":
#        table = np.column_stack((actual_M, error_array))
#        np.savetxt(outpath, table, header="M err", comments="")
#        plt.title(outpath)
#    if plot:
#        plt.xlabel("M")
#        plt.ylabel("rel. error")
#        plt.xscale("log")
#        plt.yscale("log")
#        plt.plot(actual_M, error_array)
#        plt.plot(actual_M, error_array, "x")
#        plt.show()
#    return error_array, M_array
#
#
# def curvature_estimator(u):
#    """Generate a curvature estimator"""
#
#    def error_estimate(a, c, b):
#        return np.abs(2 * u(c) - u(a) - u(b))
#
#    return error_estimate
#
#
# def min_dist_mixin(estimator, min_dist):
#    def error_estimate(a, c, b):
#        dist = b - a
#        return np.where(dist < min_dist, estimator(a, c, b), np.inf)
#
#    return error_estimate
#
#
# def find_error(
#        partitioner,
#        parameters,
#        error_function,
#        analytical_func,
#        method,
#        bc1,
#        bc2,
#        u0,
#        N,
#        t_end,
# ):
#    Ms = []
#    errors = []
#    analytical = partial(analytical_func, t=t_end)
#    for (i, param) in enumerate(parameters):
#        x = partitioner(0, 1, param)
#        Ms.append(len(x))
#        # TODO: Set to UNIFORM if x is uniform
#        grid_Mi = Grid(Grid.NON_UNIFORM, x)
#        _, U = theta_heat(
#            grid_Mi, bc1, bc2, u0, N, t_end, log=False, method=method
#        )  # solution with current M
#        errors.append(
#            error_function(U, analytical, x)
#        )
#    return Ms, errors
