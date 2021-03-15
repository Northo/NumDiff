#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from utils import *


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


def difference_norm_plot(solutions, t):
    for (i, sol) in enumerate(solutions[1:]):
        diff = solutions[i] - solutions[i - 1]
        diff_norm = discrete_l2_norm(diff)
        plt.plot(t[i], diff_norm, ".")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def find_breaking_time(solutions):
    """
    Find time of breakdown

    Criterion for breakdown:
        When solutions stops being monotonic decreasing moving from the apex
        towards either boundary.
    """
    j_end = len(solutions[0])
    for (i, sol) in enumerate(solutions):
        j_max = np.where(solutions[i] == np.amax(solutions[i]))[0][0]
        for j in range(j_max, j_end - 1):
            if solutions[i][j] < solutions[i][j + 1]:
                return t[i]
        for j in range(j_max):
            if solutions[i][j] > solutions[i][j + 1]:
                return t[i]
    else:
        print("No breakdown time found")
        return 0


######################
### System of ODEs ###
######################
x, h = np.linspace(0, 1, 1000, retstep=True)
t0 = 0
tf = 0.06

# Solve ivp/bvp
sol = solve_ivp(Fm, (t0, tf), u0(x), max_step=0.001)
t, ut = sol.t, sol.y.T

# Plot solutions at all times in [0, tf]
for (i, ti) in enumerate(t):
    plt.plot(x, ut[i], label=f"t={ti}")
#plt.legend()
plt.show()

# Difference between subsequent solutions
difference_norm_plot(ut, t)

# Find and print time of breaking
t_breaking = find_breaking_time(ut)
print(f"Time of breaking: {t_breaking}")


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
