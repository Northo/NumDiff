import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial

import numpy as np
from scipy.integrate import quad


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
    "L2 continous interpolation": lambda U, u, x: L2_continous_error(
        interpolation_continuation(U), u
    ),
}


def find_errors(M_list, f, BCs, error_functions=error_functions):
    """Find errors for given values of M for different norms
    Returns:
       errors : dict Errors for each norm, key is the norm name, value is a list
    of errors for that norm over M_list."""
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).
    errors = {error_name: [] for error_name, error_function in error_functions.items()}
    for M in M_list:
        A, F, x = generate_problem(f, M, BCs)
        U = solve_handle(A, F)
        analytical = u(BCs)
        for error_name, error_function in error_functions:
            errors[error_name].append(error_function(U, analytical, x))
    return errors


def find_errors_np(M_list, f, BCs, error_functions=error_functions):
    """Find errors for given values of M for different norms"""
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).

    errors = np.empty((len(error_functions) + 1, len(M_list)))
    errors[0, :] = M_list
    for j, M in enumerate(M_list):
        A, F, x = generate_problem(f, M, BCs)
        U = solve_handle(A, F)
        analytical = u(BCs)
        for i, (error_name, error_function) in enumerate(error_functions.items()):
            errors[i + 1, j] = error_function(U, analytical, x)

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
        M
        + 1  # Add one because range is excluding endpoint.
        + int(bc_right.is_neumann()),
    )

    # Apply values common for all BCs.
    diagonal = np.full(M + nn, -2 / h ** 2)
    upper_diagonal = np.ones(M + nn - 1) / h ** 2
    lower_diagonal = np.ones(M + nn - 1) / h ** 2
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
            warnings.warn(
                "Two Neumann conditions renders the IVP ill-posed. u(0) = 0 was imposed."
            )
    elif bc_left.is_dirichlet():
        F[0] -= bc_left.value / h ** 2
    else:
        raise Exception("Unknown boundary condition type.")

    return A, F, x[inner]


def generate_problem_variable_step(f, x, BCs=[BC(value=1), BC(value=1)]):
    """Generate probelm for variable step size x.
    Currently only implmented for Dirichlet BCs"""
    left_bc, right_bc = BCs
    if not left_bc.is_dirichlet() or not right_bc.is_dirichlet():
        raise NotImplementedError("Currently only support for Dirichlet-Dirichlet.")

    ### This is how the fuck variable step length works ###
    # There are M internal points, U_0 and U_(M+1) are on the boundary.
    # Let's say we have the x-points as below, where the number indicate
    # the index, and h the distance between.
    #
    # 0 --h0-- 1 --h1-- 2 --h2-- ... --h(M-2)-- M-1 --h(M-1)-- M --hM-- M+1
    # As written in Owren p. 69 (nice...), we have, for a point P
    # where we denote q = p+1, r = p-1:
    # r --hr-- p --hp-- q
    # U''_p = 2/(hr + hp) * [ (U_q - U_p)/hp - (U_p - U_r)/hr ]
    #       = 2/(hr + hp) * [  U_r/hr - U_p (1/hp + 1/hr) + U_q/hp ].
    ### End of how the fuck variable step length works ###

    # We will now build the "inner" part of our matrix A, the part
    # corresponding with U1 .... UM. The parts that deal with
    # BCs will be dealt with later.
    h = x[1:] - x[:-1]
    # For cleaner code, define h_back and h_front
    # At some index p, h_front gives h to next point, h_back to previous point.
    h_back, h_front = h[:-1], h[1:]
    # Note that (1/a + 1/b) / (a + b) = 1/(ab).
    diagonal = -2 / (h_back * h_front)
    upper_diagonal = 2 / (h_back + h_front) / h_front
    lower_diagonal = 2 / (h_back + h_front) / h_back
    # upper_ and lower_diagonal must be shorter than diagonal.
    # Stare hard at the slicing, it makes sense.
    A = (
        np.diag(diagonal)
        + np.diag(upper_diagonal[:-1], k=1)
        + np.diag(lower_diagonal[1:], k=-1)
    )
    F = f(x[1:-1])

    # Time to deal with BCs.
    # For Dirichelt-BCs, we do not have to change A, we simply modify F.
    # For Neumann-BCs, we extend the size of our system.
    # For two Neumann-BCs, we must be really careful, as the problem is ill-posed.

    # Right BC:
    # We move the coefficient corresponding to U_(M+1)
    # in the expression above to the other side (F).
    F[-1] -= right_bc.value * 2 / (h_front[-1] + h_back[-1]) / h_front[-1]

    # Left BC:
    F[0] -= left_bc.value * 2 / (h_front[0] + h_back[0]) / h_back[0]

    return A, F, x[1:-1]


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
        c1 = 1 / (4 * np.pi ** 2) + bc_left.value
    elif bc_left.is_neumann():
        c2 = bc_left.value
    else:
        raise Exception("Unknown boundary condition type.")

    ## Right boundary ##
    if bc_right.is_dirichlet():
        if bc_left.is_dirichlet():
            c2 = -c1 - 1 / 12 * (2 * np.pi ** 2 - 3) / np.pi ** 2 + bc_right.value
        else:
            c1 = -c2 - 1 / 12 * (2 * np.pi ** 2 - 3) / np.pi ** 2 + bc_right.value
    elif bc_right.is_neumann():
        if bc_left.is_neumann():
            warnings.warn("Ill-posed problem! Imposing u(0)=0.")
            c1 = 1 / (4 * np.pi ** 2)
        c2 = -1 / 2 + bc_right.value
    else:
        raise Exception("Unknown boundary condition type.")

    def _u(x):
        return (
            c2 * x
            + c1
            + 1
            / 12
            * (2 * np.pi ** 2 * x ** 3 - 3 * np.cos(2 * np.pi * x))
            / np.pi ** 2
        )

    return _u


def _split_interval(a, b, error_function, tol):
    """Helper function used by partition_interval"""
    c = (a + b) / 2  # Bisection
    if error_function(a, c, b) <= tol:
        partition =  [c]
    else:
        partition = [
            *_split_interval(a, c, error_function, tol),
            c,
            *_split_interval(c, b, error_function, tol)
        ]
    return partition


def partition_interval(a, b, error_function: Callable[[float, float, float], float], tol):
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
