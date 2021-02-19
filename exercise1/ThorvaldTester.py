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

N = 5
x=np.linspace(0,1, num=N+2)
x[range(0, N+2)]

M = 10
A, F, x = generate_problem(f, M, [(BCType.NEUMANN, -1), (BCType.NEUMANN, 1)])
U = np.linalg.solve(A, F)
print("AU", A @ U)
print("U", U)
plt.plot(x, U)
plt.show()
A
print(U[0]/ (x[1] - x[0]))

# +
M = 200

A, F, x = generate_problem_neumann_neumann(f, M, 0.5, 0)
first_row = np.zeros(M+2)
first_row[0] = 1
A[[0, 1], 0] = 0
U = np.linalg.solve(A, F)
plt.plot(x, U)
print((A @ U)[[0, 1]], F[[0, 1]])
print(U[[0, 1]])
print(A)
# -

M = 500
A, F, x = generate_problem(f, M, [(BCType.VALUE, 0), (BCType.NEUMANN, 0)])
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
# In other words, the problem is ill-posed.
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


def generate_problem_neumann_neumann(f, M, alpha1, alpha2):
    """Set up the matrix-problem to sovle the Poisson equation with one Neumann B.C.
    Arguments:
        f : function The function on the RHS of Poisson (u'' = f).
        M : Integer The number of internal points to use.
        alpha1/alpha2 : float Derivative position x=0 and x=1.
        
    With 'internal points' in M, one here means points in the closed interval (0,1).
    The points at x=0 and x=1 are denoted x_0 and x_(M+1), and are not
    included in the 'internal points'.
    
    As the system is only determined up to a constant, the additional constraint u(0) = 0 is applied.
    """

    x, h = np.linspace(0, 1, num=M+2, retstep=True)
    inner = range(0, M+2)
    diagonal = np.full(M+2, -2)
    upper_diagonal = np.ones(M+1)
    lower_diagonal = np.ones(M+1)
    A = np.diag(diagonal) + np.diag(upper_diagonal, k=1) + np.diag(lower_diagonal, k=-1)
    F = f(x[inner])
    
    # Right boundary condition
    F[-1] = alpha2
    A[-1, [-3, -2, -1]] = (
        np.array([1 / 2, -2, 3 / 2]) * h
    )  # Forward difference first derivative of order 2.
    
    # Left boundary condition
    F[0] = alpha1
    A[0, [0, 1, 2]] = (
        np.array([-3/2, 2, -1/2]) * h
    )  # Forward difference first derivative of order 2, setting U(0) = 0.

    return A / h**2, F, x[inner]


M = 50
a1 = 0
a2 = 0.5
A, F, x = generate_problem_neumann_neumann(f, M, a1, a2)
h = x[1] - x[0]
A, F, x


#U = np.linalg.solve(A, F)
U, res, rank, s = np.linalg.lstsq(A, F)
U -= U[0]
np.array([1/2, -2, 3/2])/h @ U[[-3, -2, -1]]

plt.plot(x, U)
plt.plot(x, f(x) * (U.max() - U.min())/2, '--', lw=0.4)

h = x[1] - x[0]
(2*U[0] - 0.5*U[1])/h

print((-2*U[0] + U[1]) / h**2, F[0])

print((U[0] - 2*U[1] + U[2])/h**2, F[1])

(-2*U[0] + U[1]) / h**2 + (2*U[0] - 0.5*U[1])/h

F[0] + h*a1


def u(x, eps=1):
    return np.exp(
    -1/eps * (x-0.5)**2
    )


plt.plot(x, u(x, eps=-1/4/np.log(10)))
plt.plot(x, f(x))

# +
B = np.vstack((
    np.zeros(52),
    A
))

B[0, [0, 1]] = [2/h, -0.5/h]
F_B = np.hstack((
a1,
    F
))
B, F_B
np.linalg.solve(B, F_B)
# -

M = 50
a1 = 0
a2 = 0.5
A, F, x = generate_problem_neumann_neumann(f, M, a1, a2)
h = x[1] - x[0]

#A[[0, 1], 0] = 0
A

A_mod = A.copy()
A_mod[[0,1], 0] = 0
U, res, rank, s = np.linalg.lstsq(A, F)
U_mod, _, _, _ = np.linalg.lstsq(A_mod, F)
U -= U[0]
plt.plot(x, U)
plt.plot(x, U_mod)


# **d)**

# +
def u(x, eps=1):
    # u(0) = exp(-1/(4*eps))
    return np.exp(-1/eps * (x-0.5)**2)
    
def f(x, eps=1):
    return 1/eps**2 * (
    u(x) * (4*x**2 - 4*x - 2*eps + 1)
    )


# -

plt.plot(x, u(x), label="u")
plt.plot(x, f(x), label="f")
plt.legend()




