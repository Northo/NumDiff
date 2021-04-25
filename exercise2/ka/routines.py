import numpy as np
from scipy.sparse import diags, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt


# Hermdog's BC class from exercise 1
class BoundaryCondition:
    DIRCHLET = 1
    NEUMANN = 2

    def __init__(self, type, value):
        self.type = type
        self.value = value
        """
        if callable(value):
            self.value = value
        else:
            self.value = lambda t: value
        """


def grid_is_uniform(x):
    left = x[:-1]
    right = x[1:]
    diff = right - left
    if max(diff) - min(diff) < 1e-6:
        return True
    return False


def theta_heat(bc1, bc2, u0, x, N, t_end, log=True, method="cn"):
    if method == "cn":
        theta = 0.5
    elif method == "be":
        theta = 1.0
    elif method == "fe":
        theta = 0.0
    else:
        raise {
            f"Invalid method specification: {method}\n Valid options are: fe, be, cn"
        }
    if not grid_is_uniform(x):
        raise ("Unsupported error, FE does not currently support non uniform grids")

    # Set up (uniform) grids and stuff
    M = len(x)
    h = x[1] - x[0]  # Assumes uniform grid
    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(x)  # initial, t = 0
    r = k / (h ** 2)
    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U
    m = M  # m will be the dimension of the system of equations, depends on BCs
    k = 0  # Index, dep. on BCs
    l = M  # Index, dep. on BCs
    if bc1.type == BoundaryCondition.DIRCHLET:
        m -= 1
        k = 1
    if bc2.type == BoundaryCondition.DIRCHLET:
        m -= 1
        l = M - 1
    # LHS matrix diagonals
    diag = np.repeat(1 + 2 * theta * r, m)
    offdiag_upper = np.repeat(-theta * r, m - 1)
    offdiag_lower = np.repeat(-theta * r, m - 1)
    # RHS matrix diagonals
    diag_b = np.repeat(1 - 2 * r * (1 - theta), m)
    offdiag_upper_b = np.repeat(r * (1 - theta), m - 1)
    offdiag_lower_b = np.repeat(r * (1 - theta), m - 1)
    # boundary value part of RHS
    b_boundary = np.zeros(m)
    if bc1.type == BoundaryCondition.NEUMANN:
        offdiag_upper[0] = -2 * theta * r
        offdiag_upper_b[0] = 2 * r * (1 - theta)
        b_boundary[0] = -2 * r * h * bc1.value
    else:
        b_boundary[0] = r * bc1.value
    if bc2.type == BoundaryCondition.NEUMANN:
        offdiag_lower[-1] = -2 * theta * r
        offdiag_lower_b[-1] = 2 * r * (1 - theta)
        b_boundary[-1] = 2 * r * h * bc2.value
    else:
        b_boundary[-1] = r * bc2.value
    # RHS matrix (sparse)
    A = csc_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
    # LHS matrix
    B = diags([diag_b, offdiag_upper_b, offdiag_lower_b], [0, 1, -1])
    # Loop through time
    for (i, ti) in enumerate(t[1:]):
        U_p = U[k:l]
        b = B @ U_p + b_boundary
        U[k:l] = spsolve(A, b)
        if log:
            solution_matrix[i + 1] = U
    if log:
        return t, U, solution_matrix
    return t, U


def forward_euler(bc1, bc2, u0, x, N, t_end, log=True):
    return theta_heat(bc1, bc2, u0, x, N, t_end, log=log, method="fe")


def backward_euler(bc1, bc2, u0, x, N, t_end, log=True):
    return theta_heat(bc1, bc2, u0, x, N, t_end, log=log, method="be")


def crank_nicolson(bc1, bc2, u0, x, N, t_end, log=True):
    return theta_heat(bc1, bc2, u0, x, N, t_end, log=log, method="cn")


def test_method(method, M, N, t_end):
    """
    Do a testrun and plot results for a numerical solver of the heat equation.
    For "veryfying" that a method works.
    Uses initial and boundary conditions from task 2a)

    Parameters:
        method : fe, be or cn
        M : number of spacial grid points
        N : number of time grid points
        t_end : end time
    """

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    # Boundary conditions
    # bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, u0(0))
    # bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, u0(1))
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    # Solve
    x = np.linspace(0, 1, M)
    t, U, solutions = theta_heat(bc1, bc2, u0, x, N, t_end, method=method)

    # Plot time samples
    num_samples = 5
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(x, solutions[j], ".", label=f"t={ti}")
    plt.title(method)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    M = 10
    N = 100
    N_FE = 10000
    test_method("cn", M, N, 0.1)
