#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

class BoundaryCondition:
    DIRICHLET = 1
    NEUMANN = 2

    def __init__(self, type, value):
        self.type = type
        self.value = value

def f(x): return np.cos(2*np.pi*x) + x

def solve(bc1, bc2, M=500):
    x, h = np.linspace(0, 1, M, retstep=True)

    b = np.array([f(x) for x in x])
    A = np.zeros((M,M))

    for m in range(1, M-1):
        # internal point
        A[m,m] = -2/h**2
        A[m,m-1] = +1/h**2
        A[m,m+1] = +1/h**2

    c1 = 0
    c2 = 0

    if bc1.type == BoundaryCondition.DIRICHLET:
        A[0,0] = 1
        b[0] = bc1.value
        c1 = 1/(4*np.pi**2) + bc1.value
    elif bc1.type == BoundaryCondition.NEUMANN:
        A[0,0] = -3/(2*h)
        A[0,1] = +2/h
        A[0,2] = -1/(2*h)
        b[0] = bc1.value
        c2 = bc1.value
    else:
        raise("Unknown boundary condition type")

    if bc2.type == BoundaryCondition.DIRICHLET:
        A[M-1,M-1] = 1
        b[-1] = bc2.value
        if bc1.type == BoundaryCondition.DIRICHLET:
            c2 = -c1 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc2.value
        elif bc1.type == BoundaryCondition.NEUMANN:
            c1 = -c2 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc2.value
    elif bc2.type == BoundaryCondition.NEUMANN:
        A[M-1,M-3] = +1/(2*h)
        A[M-1,M-2] = -2/h
        A[M-1,M-1] = +3/(2*h)
        b[-1] = bc2.value
        if bc1.type == BoundaryCondition.NEUMANN:
            print("Warning: non-unique solution, imposing extra constraint u(0) == 0")
            c1 = 1/(4*np.pi**2) # enforces u(0) == 0
            A[0,0] = 0 # apply u(0) == 0 (i.e. "remove" U_0 from first equation)
        c2 = -1/2 + bc2.value
    else:
        raise("Unknown boundary condition type")

    U = np.linalg.solve(A, b)

    print(c1, c2)

    # exact solution
    u = c2*x + c1 + 1/12*(2*np.pi**2*x**3 - 3*np.cos(2*np.pi*x))/np.pi**2

    plt.plot(x, U, label="numerical")
    plt.plot(x, u, label="analytic")
    plt.legend()
    plt.show()

bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 2)
bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
solve(bc1, bc2)
