# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tqdm
from ipywidgets import interact
from utils import *  # Local file
import tikzplotlib

seaborn.set(style="ticks", palette="Set2")
# -

# # Analytical solution
# We have
# $$
#  u''(x) = f(x) = cos(2\pi x) + x
# $$
# where
# $$
# P_1u(0) = \alpha, P_2 u(1) = \sigma, 
# $$
# and $P_1, P_2$ are either $\partial_x$ or $\mathbb{1}$.
# The latter corresponds to a Dirichlet boundary condition, while the former is a Neumann condition.
#
# $$
# \newcommand{\Bold}[1]{\mathbf{#1}}K_{2} x + K_{1} + \frac{2 \, \pi^{2} x^{3} - 3 \, \cos\left(2 \, \pi x\right)}{12 \, \pi^{2}}
# $$
# where the following holds
# \begin{align}
# a(0) &= K_{1} - \frac{1}{4 \, \pi^{2}} \\
# a(1) &= K_{1} + K_{2} + \frac{2 \, \pi^{2} - 3}{12 \, \pi^{2}} \\
# a'(0) &= K2\\
# a'(1) &= K2 + \frac12.
# \end{align}
# In our code, we formulate a set of linear equations for $K1, K2$ from the BCs.

M = 500
A, F, x = generate_problem(f, M, (BCType.VALUE, 0), (BCType.NEUMANN, 0))
U = np.linalg.solve(A, F)
plt.plot(x, U)
plt.show()

# +
M_list = np.geomspace(10, 500, 10, dtype=int)
L2_discrete_errors = []
L2_continous_errors = []
L2_continous_errors_inter = []

BCs = [(BCType.VALUE, 0), (BCType.NEUMANN, 0)]

for M in M_list:
    A, F, x = generate_problem(f, M, *BCs)
    U = np.linalg.solve(A, F)
    analytical = u(*BCs)
    L2_discrete_errors.append(L2_discrete_error(U, analytical(x)))
    L2_continous_errors.append(L2_continous_error(step_continuation(U), analytical))
    L2_continous_errors_inter.append(
        L2_continous_error(interpolation_continuation(U), analytical)
    )
# -

plt.plot(M_list, L2_discrete_errors, "o-", label="Discrete")
plt.plot(M_list, L2_continous_errors, "x-", label="Continous")
plt.plot(M_list, L2_continous_errors_inter, "x-", label="Continous (interpolation)")
# plt.plot([1e2, 1e3, 1e3, 1e2], [10e-4, 10e-6, 10e-4, 10e-4], lw=0.5, c="gray")
plt.xscale("log")
plt.yscale("log")
plt.legend()

U_func = step_continuation(U)
plt.plot(x, U_func(x))
plt.plot(x, U)


# **a)**

# +
def find_erros(M_list, BCs, f, DEBUG=False):
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).
    error_functions = [
        ["L2 discrete", lambda U, u, x: L2_discrete_error(U, u(x))],
        ["L2 continous step", lambda U, u, x: L2_continous_error(step_continuation(U), u)],
        ["L2 continous interpolation", lambda U, u, x: L2_continous_error(interpolation_continuation(U), u)]
    ]
    errors = {error[0]:[] for error in error_functions}
    
    for M in M_list:
        A, F, x = generate_problem(f, M, *BCs)
        U = solve_handle(A, F)
        analytical = u(*BCs)
        if DEBUG:
            print("Existence neumann:", existence_neumann_neumann(F, x[1]-x[0], BCs[0][1], BCs[1][1]))
            plt.plot(x, U, label="U")
            plt.plot(x, analytical(x), label="analytical")
            plt.legend()
            plt.show()
        for error_name, error_function in error_functions:
            errors[error_name].append(error_function(U, analytical, x))
            
    return errors

def existence_neumann_neumann(F, h, sigma0, sigma1):
    integral = F[0] / 2 + np.sum(F[1:-1]) + F[-1] / 2
    integral *= h
    # The condition is that difference is zero
    difference = (sigma1 - sigma0) - integral
    return difference

# Boundary conditions.
BCs = [(BCType.VALUE, 0), (BCType.NEUMANN, 0)]

# M-values to check for.
M_list = np.geomspace(
    10,
    500,
    10,
    dtype=int
)

errors = find_erros(M_list, BCs, f) 
for name, error in errors.items():
    plt.plot(M_list, error, '-x', label=name)

plt.title("Errors as a function of points $M$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$M$")
plt.ylabel("Error")
plt.legend()
tikzplotlib.save("figures/a_error.pgf")
plt.plot()
# -

# **b)**

# +
# Boundary conditions.
BCs = [(BCType.VALUE, 0), (BCType.VALUE, 1)]

# M-values to check for.
M_list = np.geomspace(
    10,
    1000,
    10,
    dtype=int
)

errors = find_erros(M_list, BCs, f) 
for name, error in errors.items():
    plt.plot(M_list, error, '-x', label=name)

plt.title("Errors as a function of points $M$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$M$")
plt.ylabel("Error")
plt.legend()
tikzplotlib.save("figures/a_error.pgf")
plt.plot()
# -

# ## The issue of two Neumann-conditions
# When having two Neumann conditions, the solution is only specified up to a constant, as a constant is the homogeneous solution to Poisson's equation.
# We solve this by simply adding the constraint $u(0) = 0$.

# +
# Boundary conditions.
alpha1 = 0
alpha2 = 0.5
BCs = [(BCType.NEUMANN, alpha1), (BCType.NEUMANN, alpha2)]


def find_erros(M_list, BCs, f, DEBUG=False):
    # Error functions to measure.
    # Each function must have the call signature f(U:ndarray, u:function, x:ndarray).
    error_functions = [
        ["L2 discrete", lambda U, u, x: L2_discrete_error(U, u(x))],
        ["L2 continous step", lambda U, u, x: L2_continous_error(step_continuation(U), u)],
        ["L2 continous interpolation", lambda U, u, x: L2_continous_error(interpolation_continuation(U), u)]
    ]
    errors = {error[0]:[] for error in error_functions}
    
    for M in M_list:
        A, F, x = generate_problem_neumann_neumann(f, M, alpha1, alpha2)
        U = solve_handle(A, F)
        analytical_nonzero = u(*BCs)
        analytical = lambda x: analytical_nonzero(x) - analytical_nonzero(0)
        if DEBUG:
            print("Existence neumann:", existence_neumann_neumann(F, x[1]-x[0], BCs[0][1], BCs[1][1]))
            plt.plot(x, U, label="U")
            plt.plot(x, analytical(x), label="analytical")
            plt.legend()
            plt.show()
        for error_name, error_function in error_functions:
            errors[error_name].append(error_function(U, analytical, x))
            
# M-values to check for.
M_list = np.geomspace(
    20,
    5000,
    10,
    dtype=int
)

errors = find_erros(M_list, BCs, f, DEBUG=True) 
for name, error in errors.items():
    plt.plot(M_list, error, '-x', label=name)

plt.title("Errors as a function of points $M$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$M$")
plt.ylabel("Error")
plt.legend()
tikzplotlib.save("figures/a_error.pgf")
plt.plot()
# -
# ## 





