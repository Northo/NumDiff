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
        if callable(value):
            self.value = value
        else:
            self.value = lambda t: value


class Grid:
    UNIFORM = 1
    NON_UNIFORM = 2

    def __init__(self, type, x):
        self.type = type
        if type == Grid.UNIFORM:
            self.is_uniform = True
        else:
            self.is_uniform = False
        self.x = x
        self.h = x[1:] - x[:-1]


def forward_euler(grid, bc1, bc2, u0, N, t_end, log=True):
    """
    Solve the 1D heat equation using forward Euler method

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """
    if not grid.is_uniform:
        raise ("Unsupported error, FE does not currently support non uniform grids")
    x = grid.x
    M = len(x)
    h = grid.h[0]  # Uniform grid, all equal
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    for (i, ti) in enumerate(t[1:]):
        if bc1.type == BoundaryCondition.DIRCHLET:
            U[0] = bc1.value(ti)
        elif bc1.type == BoundaryCondition.NEUMANN:
            U[0] += r * (U[1] - U[0] - 2 * h * bc1.value(ti))
        else:
            raise ("Unsupported boundary condition type")
        if bc2.type == BoundaryCondition.DIRCHLET:
            U[-1] = bc2.value(ti)
        elif bc2.type == BoundaryCondition.NEUMANN:
            U[-1] += r * (U[-2] - U[-1] + 2 * h * bc1.value(ti))
        else:
            raise ("Unsupported boundary condition type")
        U[1:-1] = U[1:-1] + r * (U[:-2] - 2 * U[1:-1] + U[2:])
        if log:
            solution_matrix[i + 1] = U
    if log:
        return t, U, solution_matrix
    return t, U


def backward_euler(grid, bc1, bc2, u0, N, t_end, log=True):
    """
    Solve the 1D heat equation using backward Euler method

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """

    x = grid.x
    M = len(x)
    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(x)  # initial, t = 0
    h = grid.h[0]
    r = k / (h ** 2)

    h_arr = grid.h  # array
    h_left, h_right = h_arr[:-1], h_arr[1:]
    r_arr = 2 * k / (h_left + h_right)

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
        if not grid.is_uniform:
            raise (
                "Unsupported error, does not currently support non uniform grids for these BCs"
            )
        # neumann-neumann
        diag = np.repeat(1 + 2 * r, M)
        offdiag_upper = np.repeat(-r, M - 1)
        offdiag_lower = np.repeat(-r, M - 1)
        diag[0] = 1 - r
        offdiag_upper[0] = r
        diag[-1] = 1 + r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U
            b[0] -= r * 2 * r * h * bc1.value(ti)
            b[-1] += r * 2 * r * h * bc2.value(ti)
            U = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # dirchlet-dirchlet
        m = M - 2
        diag = 1 + r_arr * (1 / h_left + 1 / h_right)
        offdiag_upper = -r_arr[:-1] / h_right[:-1]
        offdiag_lower = -r_arr[1:] / h_left[1:]
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))

        for (i, ti) in enumerate(t[1:]):
            b = U[1:-1]
            b[0] += k / (h_arr[0] ** 2) * bc1.value(ti)
            b[-1] += k / (h_arr[-1] ** 2) * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
        """
        m = M - 2
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[1:-1]
            b[0] += r * bc1.value(ti)
            b[-1] += r * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
        """
    elif (
        bc1.type == BoundaryCondition.DIRCHLET and bc2.type == BoundaryCondition.NEUMANN
    ):
        # dirchlet-neumann
        if not grid.is_uniform:
            raise (
                "Unsupported error, does not currently support non uniform grids for these BCs"
            )
        m = M - 1
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        diag[-1] = 1 + r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[1:]
            b[0] += r * bc1.value(ti)
            b[-1] += r * 2 * r * h * bc2.value(ti)
            U[1:] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    elif (
        bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # neumann-dirchlet
        if not grid.is_uniform:
            raise (
                "Unsupported error, does not currently support non uniform grids for these BCs"
            )
        m = M - 1
        diag = np.repeat(1 + 2 * r, m)
        offdiag_upper = np.repeat(-r, m - 1)
        offdiag_lower = np.repeat(-r, m - 1)
        diag[0] = 1 - r
        offdiag_upper[0] = r
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = U[:-1]
            b[0] -= r * 2 * r * h * bc1.value(ti)
            b[-1] += r * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return t, U, solution_matrix
    return t, U


def crank_nicolson(grid, bc1, bc2, u0, N, t_end, log=True):
    """
    Solve the 1D heat equation using backward Crank-Nicolson

    Parameters:
        bc1 : BoundaryCondition at x=0
        bc2 : BoundaryCondition at x=1
        M : Number of spacial grid points
        N : Number of time grid points
        t_end : ending time for computation/iteration
    Returns:
        x : spatial grid
        t : time grid
        U : solution of the heat equation at time t_end
        solution_matrix : NxM matrix, row i is U at time ti 0 < ti < t_end
    """
    x = grid.x
    M = len(x)
    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(x)  # initial, t = 0
    h = grid.h[0]
    r = k / (h ** 2)

    h_arr = grid.h  # array
    h_left, h_right = h_arr[:-1], h_arr[1:]
    r_arr = 2 * k / (h_left + h_right)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
        # Neuman-Neuman
        diag = np.repeat(1 + r, M)
        offdiag_upper = np.repeat(-r / 2, M - 1)
        offdiag_lower = np.repeat(-r / 2, M - 1)
        diag[0] = 1 + r / 2
        offdiag_upper[0] = -r / 2
        diag[-1] = 1 + r / 2
        offdiag_lower[-1] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = np.zeros(M)
            b[1:-1] = (r / 2) * U[:-2] + (1 - r) * U[1:-1] + (r / 2) * U[2:]
            b[0] = U[0] + (r / 2) * (U[1] - U[0]) - 2 * r * h * bc1.value(ti)
            b[-1] = U[-1] + (r / 2) * (U[-2] - U[-1]) + 2 * r * h * bc2.value(ti)
            U = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # DIRCHLET-DIRCHLET
        m = M - 2
        diag = 1 + r_arr/2 * (1 / h_left + 1 / h_right)
        offdiag_upper = -r_arr[:-1] / (2 * h_right[:-1])
        offdiag_lower = -r_arr[1:] / (2 * h_left[1:])
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))

        for (i, ti) in enumerate(t[1:]):
            b = U[1:-1] + r_arr/2 * (U[2:]/h_right - (1/h_left + 1/h_right)*U[1:-1] + U[:-2]/h_left)
            b[0] += k / (2*h_arr[0] ** 2) * bc1.value(ti)
            b[-1] += k / (2*h_arr[-1] ** 2) * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
        """
        m = M - 2
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-2] + (1 - r) * U[1:-1] + (r / 2) * U[2:]
            b[0] += (r / 2) * bc1.value(ti)
            b[-1] += (r / 2) * bc2.value(ti)
            U[1:-1] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
        """
    elif (
        bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # NEUMANN-DIRCHLET
        print("WARNING: Case Neumann-Dirchlet not yet tested, werid stuff may happen!")
        m = M - 1
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        diag[0] = 1 + r / 2
        offdiag_upper[0] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-1] + (1 - r) * U[:-1] + (r / 2) * U[1:]
            b[0] += (r / 2) * (U[1] - U[0]) - 2 * r * h * bc1.value(ti)
            b[-1] += (r / 2) * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET and bc2.type == BoundaryCondition.NEUMANN
    ):
        # DIRCHLET-NEUMANN
        print("WARNING: Case Dirchlet-Neumann not yet tested, werid stuff may happen!")
        m = M - 1
        diag = np.repeat(1 + r, m)
        offdiag_upper = np.repeat(-r / 2, m - 1)
        offdiag_lower = np.repeat(-r / 2, m - 1)
        diag[-1] = 1 + r / 2
        offdiag_lower[-1] = -r / 2
        A = csr_matrix(diags([diag, offdiag_upper, offdiag_lower], [0, 1, -1]))
        for (i, ti) in enumerate(t[1:]):
            b = (r / 2) * U[:-1] + (1 - r) * U[1:] + (r / 2) * U[1:]
            b[0] += (r / 2) * bc1.value(ti)
            b[-1] += (r / 2) * (U[-2] - U[-1]) + 2 * r * h * bc2.value(ti)
            U[1:] = spsolve(A, b)
            if log:
                solution_matrix[i + 1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return t, U, solution_matrix
    return t, U


def test_method(method, grid, N, t_end):
    """
    Do a testrun and plot results for a numerical solver of the heat equation.
    For "veryfying" that a method works.
    Uses initial and boundary conditions from task 2a)

    Parameters:
        method : the function name of the method to be tested e.g. forward_euler
        M : number of spacial grid points
        N : number of time grid points
        t_end : end time
    """

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, u0(0))
    bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, u0(1))
    # bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    # bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    t, U_final, solutions = method(grid, bc1, bc2, u0, N, t_end)

    num_samples = 5
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(grid.x, solutions[j], ".", label=f"t={ti}")
    plt.title(method.__name__)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    M = 100
    N = 100
    N_FE = 10000
    grid = Grid(Grid.NON_UNIFORM, np.linspace(0, 1, M) ** 2)
    unigrid = Grid(Grid.UNIFORM, np.linspace(0, 1, M))
    # test_method(forward_euler, unigrid, N_FE, 0.1)
    test_method(backward_euler, unigrid, N, 0.1)
    test_method(backward_euler, grid, N, 0.1)
    test_method(crank_nicolson, unigrid, N, 0.1)
    test_method(crank_nicolson, grid, N, 0.1)
