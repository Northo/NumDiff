# Numerical routines for solving the heat equation
import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from utils import BoundaryCondition, Grid


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

    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(grid.x)  # initial, t = 0
    M = len(grid.x)

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if grid.is_uniform:
        r = k / (grid.h ** 2)
        for (i, ti) in enumerate(t[1:]):
            if bc1.type == BoundaryCondition.DIRCHLET:
                U[0] = bc1.value(ti)
            elif bc1.type == BoundaryCondition.NEUMANN:
                U[0] += r * (U[1] - U[0] - 2 * grid.h * bc1.value(ti))
            else:
                raise ("Unsupported boundary condition type")
            if bc2.type == BoundaryCondition.DIRCHLET:
                U[-1] = bc2.value(ti)
            elif bc2.type == BoundaryCondition.NEUMANN:
                U[-1] += r * (U[-2] - U[-1] + 2 * grid.h * bc1.value(ti))
            else:
                raise ("Unsupported boundary condition type")
            U[1:-1] = U[1:-1] + r * (U[:-2] - 2 * U[1:-1] + U[2:])
            if log:
                solution_matrix[i+1] = U
    else:
        raise("Adaptive grid not yet implemented!")
    if log:
        return grid.x, t, U, solution_matrix
    return grid.x, t, U


def bacward_euler_uniform_grid(grid, M, t, k, bc1, bc2, U, solution_matrix, log):
    r = k / (grid.h ** 2)
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
            b[0] -= r * 2 * r * grid.h * bc1.value(ti)
            b[-1] += r * 2 * r * grid.h * bc2.value(ti)
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
            b[-1] += r * 2 * r * grid.h * bc2.value(ti)
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
            b[0] -= r * 2 * r * grid.h * bc1.value(ti)
            b[-1] += r * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
    )


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

    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(grid.x)  # initial, t = 0
    M = len(grid.x)

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if grid.is_uniform:
        if log:
            bacward_euler_uniform_grid(grid, M, t, k, bc1, bc2, U, solution_matrix, log)
        else:
            bacward_euler_uniform_grid(grid, M, t, k, bc1, bc2, U, 0, log)
    else:
        raise("Adaptive grid not yet implemented")
    if log:
        return grid.x, t, U, solution_matrix
    return grid.x, t, U



def crank_nicolson_uniform_grid(grid, M, t, k, bc1, bc2, U, solution_matrix, log):
    r = k / (grid.h ** 2)
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
            b[0] = U[0] + (r / 2) * (U[1] - U[0]) - 2 * r * grid.h * bc1.value(ti)
            b[-1] = U[-1] + (r / 2) * (U[-2] - U[-1]) + 2 * r * grid.h * bc2.value(ti)
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
            b[0] += (r / 2) * (U[1] - U[0]) - 2 * r * grid.h * bc1.value(ti)
            b[-1] += (r / 2) * bc2.value(ti)
            U[:-1] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
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
            b[-1] += (r / 2) * (U[-2] - U[-1]) + 2 * r * grid.h * bc2.value(ti)
            U[1:] = spsolve(A, b)
            if log:
                solution_matrix[i+1] = U
    else:
        raise (
            "Unsupported boundary condition type(s). The supported are: Dirchlet and Neumann."
        )


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

    t, k = np.linspace(0, t_end, N, retstep=True)
    U = u0(grid.x)  # initial, t = 0
    M = len(grid.x)

    if log:
        solution_matrix = np.zeros((N, M))
        solution_matrix[0] = U

    if grid.is_uniform:
        if log:
            crank_nicolson_uniform_grid(grid, M, t, k, bc1, bc2, U, solution_matrix, log)
        else:
            crank_nicolson_uniform_grid(grid, M, t, k, bc1, bc2, U, 0, log)
    else:
        raise("Adaptiveness not yet implemented")
    if log:
        return grid.x, t, U, solution_matrix
    return grid.x, t, U
