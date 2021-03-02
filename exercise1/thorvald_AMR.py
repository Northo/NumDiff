from utils import *
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

def u(x, eps=1):
    return np.exp(-1/eps * (x-0.5)**2)


def f(x, eps=1):
    return -2 * u(x, eps=eps) * (eps - 2 * (0.5 - x)**2) / eps**2


############
# Solve the 


M = 100
eps = 0.05

f = partial(f, eps=eps)
u = partial(u, eps=eps)
x = np.linspace(0, 1, M)

error = lambda a, c, b: np.abs(f(c) * (b-c))
error_exact = lambda a, c, b: np.abs(
    (b-a) * (f(c) - (u(a) + u(b) - 2*u(c)))
)

def solve_for_error(error_function, tol):
    x = partition_interval(0, 1, error_function, tol)
    A, F, x = generate_problem_variable_step(f, x, BCs=[BC(value=0), BC(value=0)])
    U = np.linalg.solve(A, F)
    return x, U
x1, U1 = solve_for_error(error, 0.5)
x2, U2 = solve_for_error(error_exact, 0.9)

plt.plot(x1, U1, ls="dashed", label=f"Num {len(x1)}pts")
plt.plot(x2, U2, ls="dashdot", label=f"num2 {len(x2)}pts")
plt.plot(x1, u(x1), alpha=0.5, label="Anal")

plt.scatter(x1, np.zeros_like(x1), marker="|")
plt.scatter(x2, np.zeros_like(x2)+0.06, marker="|")

#plt.plot(x, np.abs(U - u(x))*100, '-x', label="Error x100")
plt.legend()
plt.show()
