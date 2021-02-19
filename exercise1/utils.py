from enum import Enum
from functools import partial
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


import warnings

class BCType(Enum):
    DIRICHLET = 1
    NEUMANN = 2
    VALUE = 1  # Alias for Dirichlet.

@dataclass
class BC:
    type: BCType = BCType.DIRICHLET
    value: float = 0

    def is_neumann(self):
        return self.type == BCType.NEUMANN

    def is_dirichlet(self):
        return self.type == BCType.DIRICHLET

DEFAULT_BCs = [BC(BCType.VALUE, 0), (BCType.NEUMANN, 1)]


def L2_discrete_norm(V):
    """Calculate the L2 discrete norm.
    Arguments:
        V : ndarray The vector for which we find the norm."""
    return np.linalg.norm(V) / np.sqrt(len(V))


def L2_continous_norm(v, x_min=0, x_max=1):
    """Calculate the L2 continous norm.
    Arguments:
        v : function The function for which we take the norm."""
    integrand = lambda x: v(x) ** 2
    return np.sqrt(quad(integrand, x_min, x_max)[0])  # quad from scipy.integrate


def L2_discrete_error(U, u):
    """Calculate the L2 discrete error of U w.r.t. u.
    Arguments:
        U : ndarray The array for which we find the error.
        u : ndarray The correct array, used as reference."""
    return L2_discrete_norm(u - U) / L2_discrete_norm(u)


def L2_continous_error(U, u):
    """Calculate the L2 continous error of U w.r.t. u.
    Arguments:
        U : function The function for which we find the error.
        u : function The correct function, used as reference."""
    return L2_continous_norm(lambda x: u(x) - U(x)) / L2_continous_norm(u)


def step_continuation(V, x_min=0, x_max=1):
    """Generates a function which is the stepwise continuation of the array V.
    Example:
    Given the array [1, 2, 4, 5, 3], and with x_min=0 and x_max=1, we return a function
    which is 1 between 0 and 0.2, 2 between 0.2 and 0.4 and so forth."""
    N = len(V)

    @np.vectorize
    def V_continous(x):
        i = np.trunc(np.clip(N * x / (x_max - x_min), 0, N - 1))
        return V[int(i)]

    return V_continous


def interpolation_continuation(V, x_min=0, x_max=1):
    """Generates a function that is the interpolated continuation of the array V"""
    N = len(V)
    x_space = np.linspace(x_min, x_max, N)

    def V_continous(x):
        return np.interp(x, x_space, V)

    return V_continous

error_functions = {
    "L2 discrete": lambda U, u, x: L2_discrete_error(U, u(x)),
    "L2 continous step": lambda U, u, x: L2_continous_error(step_continuation(U), u),
    "L2 continous interpolation": lambda U, u, x: L2_continous_error(interpolation_continuation(U), u)
}


def find_errors(M_list, f, BCs, error_functions=error_functions):
    """Find errors for given values of M for different norms
    Returns:
       errors : dict Errors for each norm, key is the norm name, value is a list
    of errors for that norm over M_list."""
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).
    errors = {error_name:[] for error_name, error_function in error_functions.items()}
    for M in M_list:
        A, F, x = generate_problem(f, M, BCs)
        U = solve_handle(A, F)
        analytical = u(BCs)
        for error_name, error_function in error_functions:
            errors[error_name].append(
                error_function(U, analytical, x)
            )
    return errors

def find_errors_np(M_list, f, BCs, error_functions=error_functions):
    """Find errors for given values of M for different norms"""
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).

    errors = np.empty((len(error_functions)+1, len(M_list)))
    errors[0, :] = M_list
    for j, M in enumerate(M_list):
        A, F, x = generate_problem(f, M, BCs)
        U = solve_handle(A, F)
        analytical = u(BCs)
        for i, (error_name, error_function) in enumerate(error_functions.items()):
            errors[i+1, j] = error_function(U, analytical, x)

    return errors

def write_errors_file(filename, Ms, errors):
    """Write errors defined in the format returned by find_erros to a file"""
    with open(filename, "w") as file:
        file.write("M\t")
        for error_name in errors:
            file.write(error_name + "\t")
        for i in range(len(Ms)):
            file.write("\n")
            file.write(str(Ms[i]))
            for error in errors:
                file.write("\t" + f"{errors[error][i]:8.3f}")

def existence_neumann_neumann(F, h, sigma0, sigma1):
    """Condition of existence for a neumann-neumann problem, discrete formulation."""
    integral = F[0] / 2 + np.sum(F[1:-1]) + F[-1] / 2
    integral *= h
    # The condition is that difference is zero
    difference = (sigma1 - sigma0) - integral
    return difference == 0


def generate_problem(f, M, BCs=DEFAULT_BCs):
    """Set up the matrix-problem to sovle the Poisson equation with one Neumann B.C.
    Arguments:
        f : function The function on the RHS of Poisson (u'' = f).
        M : Integer The number of internal points to use.
        bc_left/right: tuple The boundary conditions of the problem. The tuple has two,
                       elements, where the first defines the type and the second the value.

    With 'internal points' in M, one here means points in the closed interval (0,1).
    The points at x=0 and x=1 are denoted x_0 and x_(M+1), and are not
    included in the 'internal points'."""

    bc_left, bc_right = BCs

    # Depending on the BCs, the size of our matrix varies.
    # We need at least an MxM-system. For each Neumann-BC we must increase by one.
    # nn: number_neumann, for brevity.
    nn = np.sum([BC.is_neumann() for BC in [bc_left, bc_right]])
    # Independent of nn, we want M+2 points in order to be consistent with notation in problem text.
    x, h = np.linspace(0, 1, num=M + 2, retstep=True)
    # For later convenience, we will define inner, which is simply the range of x-values we actually use.
    inner = range(
        int(bc_left.is_dirichlet()),
        M + 1  # Add one because range is excluding endpoint.
        + int(
            bc_right.is_neumann()
        ),
    )

    # Apply values common for all BCs.
    diagonal = np.full(M + nn, -2/h**2)
    upper_diagonal = np.ones(M + nn - 1)/h**2
    lower_diagonal = np.ones(M + nn - 1)/h**2
    A = np.diag(diagonal) + np.diag(upper_diagonal, k=1) + np.diag(lower_diagonal, k=-1)
    F = f(x[inner])

    # Change elements specific for BCs.
    if bc_right.is_neumann():
        F[-1] = bc_right.value
        A[-1, [-3, -2, -1]] = (
            np.array([1 / 2, -2, 3 / 2]) / h
        )  # Forward difference first derivative of order 2.
    elif bc_right.is_dirichlet():
        F[-1] -= bc_right.value / h ** 2
    else:
        raise Exception("Unknown boundary condition type.")

    if bc_left.is_neumann():
        F[0] = bc_left.value
        A[0, [0, 1, 2]] = (
            np.array([-3 / 2, 2, -1 / 2]) * h
        )  # Forward difference first derivative of order 2.
        if bc_right.is_neumann():
            # Solution only determined up to a constant.
            # We set u(0) = 0 as extra condition.
            A[:, 0] = 0
            warnings.warn("Two Neumann conditions renders the IVP ill-posed. u(0) = 0 was imposed.")
    elif bc_left.is_dirichlet():
        F[0] -= bc_left.value / h ** 2
    else:
        raise Exception("Unknown boundary condition type.")

    return A, F, x[inner]


def f(x):
    return np.cos(2 * np.pi * x) + x


def solve_handle(A, F, retresidual=False, retrank=False, rets=False):
    """Solve the system Ax=F, but handle singular."""
    try:
        U = np.linalg.solve(A, F)
        residual, rank, s = None, None, None
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix, least square used.")
        U, residual, rank, s = np.linalg.lstsq(A, F)
    stats = []
    if retresidual:
        stats.append(residual)
    if retrank:
        stats.append(rank)
    if rets:
        stats.append(s)
    if stats:
        return U, *stats
    else:
        return U


def u(BCs=DEFAULT_BCs):
    """Much nicer analytical solution finder taken from Herman."""
    c1 = c2 = 0
    bc_left, bc_right = BCs

    ## Left bounary ##
    if bc_left.is_dirichlet():
        c1 = 1/(4 * np.pi**2) + bc_left.value
    elif bc_left.is_neumann():
        c2 = bc_left.value
    else:
        raise Exception("Unknown boundary condition type.")

    ## Right boundary ##
    if bc_right.is_dirichlet():
        if bc_left.is_dirichlet():
            c2 = -c1 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc_right.value
        else:
            c1 = -c2 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc_right.value
    elif bc_right.is_neumann():
        if bc_left.is_neumann():
            warnings.warn("Ill-posed problem! Imposing u(0)=0.")
            c1 = 1 / (4 * np.pi**2)
        c2 = -1/2 + bc_right.value
    else:
        raise Exception("Unknown boundary condition type.")


    def _u(x):
        return c2*x + c1 + 1/12*(2*np.pi**2*x**3 - 3*np.cos(2*np.pi*x))/np.pi**2
    return _u


# def u(BCs=DEFAULT_BCs):
#     """Returns the analytical solution with given BCs.
#     See analytical solution to understand spaghetti."""

#     # We wish to find K1 and K2 so that our BCs are solved.
#     # This equates to solving two linear equations, one from each BC.
#     bc_left, bc_right = BCs
#     if bc_left.type == BCType.VALUE:
#         bc_left_eq = (1, 0, 1 / (4 * np.pi ** 2) + bc_left.value)
#     else:
#         bc_left_eq = (0, 1, bc_left.value)

#     if bc_right.type == BCType.VALUE:
#         bc_right_eq = (1, 1, -(2 * np.pi ** 2 - 3) / (12 * np.pi ** 2) + bc_right.value)
#     else:
#         bc_right_eq = (0, 1, -0.5 + bc_right.value)

#     A = np.empty((2, 2))
#     A[0, :] = bc_left_eq[:2]
#     A[1, :] = bc_right_eq[:2]
#     b = np.empty(2)
#     b[0] = bc_left_eq[2]
#     b[1] = bc_right_eq[2]
#     K1, K2 = solve_handle(A, b)

#     def _u(x):
#         return (
#             K2 * x
#             + K1
#             + (2 * np.pi ** 2 * x ** 3 - 3 * np.cos(2 * np.pi * x)) / (12 * np.pi ** 2)
#         )

#     return np.vectorize(_u)
