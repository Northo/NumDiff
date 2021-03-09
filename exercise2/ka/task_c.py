#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def u0(x):
    """ Initial condition """
    return np.exp(-400 * (x - 1 / 2) ** 2)


def Fm(t, v):
    """ RHS of system of ODEs """
    M = len(v)
    h = 1 / (M - 1)
    Fm = np.empty(M)
    Fm[0] = 0  # Hardcoded BC
    Fm[-1] = 0  # Hardcoded BC
    Fm[1:-1] = -v[1:-1] * (1 / (2 * h)) * (v[2:] - v[:-2])  # Inner points
    # Fm[1:-1] = -1/(4*h) * (v[2:]**2 - v[:-2]**2) # Inner points
    return Fm


def lax_friedrich(u0, N, M, t_end, log=True):
    """ Probably not quite right, don't use this """
    x, h = np.linspace(0, 1, M, retstep=True)
    t, k = np.linspace(0, t_end, retstep=True)
    U = u0(x)  # Initial

    if log:
        solution_matrix = np.empty((N, M))
        solution_matrix[0] = U

    for (i, ti) in enumerate(t):
        U[1:-1] = 1 / 2 * (U[2:] + U[:-2]) - k / (2 * h) * (
            U[2:] - U[:-2]
        )  # Pretty sure there is an error in here
        if log:
            solution_matrix[i + 1] = U
    if log:
        return t, x, U, solution_matrix
    return t, x, U


######################
### System of ODEs ###
######################
x, h = np.linspace(0, 1, 1000, retstep=True)
t0 = 0
tf = 0.1

# Solve ivp/bvp
sol = solve_ivp(Fm, (t0, tf), u0(x), max_step=0.001)
t, ut = sol.t, sol.y.T
print(sol.success)

for (i, ti) in enumerate(t):
    plt.plot(x, ut[i], label=f"t={ti}")
plt.legend()
plt.show()


########################################
### Lax-Friedrich (1st order method) ###
########################################
# M = 100
# N = 100
# t_end = 0.5
#
# t, x, U, sols = lax_friedrich(u0, N, M, t_end)
# for (i, ti) in enumerate(t):
#    plt.plot(x, sols[i], label=f"t={ti}")
# plt.legend()
# plt.show()
# plt.show()
