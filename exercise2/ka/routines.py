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
    ADAPTIVE = 2

    def __init__(self, type, ):
        self.type = type


def forward_euler(bc1, bc2, u0, M, N, t_end, log=True):
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

    x, h = np.linspace(0, 1, M, retstep=True)
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
            solution_matrix[i+1] = U
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def backward_euler(bc1, bc2, u0, M, N, t_end, log=True):
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

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
    U = u0(x)  # initial, t = 0

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
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
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # dirchlet-dirchlet
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
    elif (
        bc1.type == BoundaryCondition.DIRCHLET and bc2.type == BoundaryCondition.NEUMANN
    ):
        # dirchlet-neumann
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
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # neumann-dirchlet
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
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def crank_nicolson(bc1, bc2, u0, M, N, t_end, log=True):
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

    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, N, retstep=True)
    r = k / (h ** 2)
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
                solution_matrix[i+1] = U
    elif (
        bc1.type == BoundaryCondition.DIRCHLET
        and bc2.type == BoundaryCondition.DIRCHLET
    ):
        # DIRCHLET-DIRCHLET
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
                solution_matrix[i+1] = U
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
                solution_matrix[i+1] = U
        if log:
            return x, t, U, solution_matrix
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
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )
    if log:
        return x, t, U, solution_matrix
    return x, t, U


def test_method(method, M, N, t_end):
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

    # bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, initial(0))
    # bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, initial(1))
    bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
    bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)

    def u0(x):
        """ Initial condition u(x, 0) = 2*pi*x - sin(2*pi*x) """

        return 2 * np.pi * x - np.sin(2 * np.pi * x)

    x, t, U_final, solutions = method(bc1, bc2, u0, M, N, t_end)

    num_samples = 5
    for i in range(num_samples):
        j = i * N // num_samples
        ti = t[j]
        plt.plot(x, solutions[j], ".", label=f"t={ti}")
    plt.title(method.__name__)
    plt.legend()
    plt.show()
