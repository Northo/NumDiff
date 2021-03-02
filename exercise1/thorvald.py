from utils import *
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data_thorvald/"  # Relative path for data files.
DEBUG = False
DEBUG_C = True

#######################
# Common for a) to c) #
#######################
Ms = np.geomspace(20, 500, 10, dtype=int)
def f(x):
    return np.cos(2 * np.pi * x) + x

def do_BC(BCs, filename):
    u = get_u(BCs)
    errors = find_errors_M(Ms, f, u, BCs)
    write_errors_file(DATA_PATH + filename, Ms, errors)
    if DEBUG:
        plot_errors(errors, Ms)
        plt.legend()
        plt.show()
    return errors

# a) ##########
BCs = [
    BC(),
    BC(BCType.NEUMANN, 0)
]
do_BC(BCs, "a.dat")

# b) ##########
BCs = [
    BC(BCType.DIRICHLET, 1),
    BC(BCType.DIRICHLET, 1)
]
do_BC(BCs, "b.dat")


# c) ##########
BCs = [
    BC(BCType.NEUMANN, 0),
    BC(BCType.NEUMANN, 0.5)
]
do_BC(BCs, "c.dat")


# d) ##########
def u(x, eps=1):
    """Given manufactured solution."""
    return np.exp(-1/eps * (x-0.5)**2)


def f(x, eps=1):
    """"Analytical solution."""
    return -2 * u(x, eps=eps) * (eps - 2 * (0.5 - x)**2) / eps**2

BCs = [
    BC(BCType.DIRICHLET, u(0)),
    BC(BCType.DIRICHLET, u(1))
]
tols = np.linspace(0.05, 0.9, 10)

errors = find_errors_tol(tols, f, u, BCs)
if DEBUG or DEBUG_C:
    plot_errors(errors, tols)
    plt.legend()
    plt.show()
