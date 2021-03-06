from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import partial

DATA_PATH = "data_thorvald/"  # Relative path for data files.
DEBUG = False
DEBUG_C = True

# #######################
# # Common for a) to c) #
# #######################
# Ms = np.geomspace(20, 500, 10, dtype=int)
# def f(x):
#     return np.cos(2 * np.pi * x) + x

# def do_BC(BCs, filename):
#     u = get_u(BCs)
#     errors = find_errors(Ms, f, u, BCs)
#     write_errors_file(DATA_PATH + filename, Ms, errors)
#     if DEBUG:
#         plot_errors(errors, Ms)
#         plt.legend()
#         plt.show()
#     return errors

# # a) ##########
# BCs = [
#     BC(),
#     BC(BCType.NEUMANN, 0)
# ]
# do_BC(BCs, "a.dat")

# # b) ##########
# BCs = [
#     BC(BCType.DIRICHLET, 1),
#     BC(BCType.DIRICHLET, 1)
# ]
# do_BC(BCs, "b.dat")


# # c) ##########
# BCs = [
#     BC(BCType.NEUMANN, 0),
#     BC(BCType.NEUMANN, 0.5)
# ]
# do_BC(BCs, "c.dat")


# d) ##########
def u(x, eps=1):
    """Given manufactured solution."""
    return np.exp(-1 / eps * (x - 0.5) ** 2)


def f(x, eps=1):
    """"Analytical solution."""
    return -2 * u(x, eps=eps) * (eps - 2 * (0.5 - x) ** 2) / eps ** 2


eps = 0.01
u = partial(u, eps=eps)
f = partial(f, eps=eps)

BCs = [BC(BCType.DIRICHLET, u(0)), BC(BCType.DIRICHLET, u(1))]
tols = np.linspace(0.01, 0.5, 30)

#### Norms ####
disc = DEFAULT_ERROR_FUNCTIONS["L2 discrete"]
step = DEFAULT_ERROR_FUNCTIONS["L2 continous step"]
cont = DEFAULT_ERROR_FUNCTIONS["L2 continous interpolation"]
#### Partitioners ####
AMR_f = partial(
    partition_interval, error_function=min_dist_mixin(quad_curvature_estimator(f), 0.3)
)
AMR_exact = partial(partition_interval, error_function=exact_estimator(f, u))

methods = [
    # Partitioner, parameters (M or tol), norm, label
    [np.linspace, [int(f) for f in np.geomspace(10, 500, 10)], disc, "UMR discrete"],
    [AMR_f, np.linspace(20, 0.1, 10), disc, "AMR_f discrete"],
]

errors = []
for partitioner, parameters, norm, label in methods:
    if DEBUG or DEBUG_C:
        print("Running series", label)
    errors.append({"errors": [], "Ms": [], "label": label})
    for parameter in parameters:
        x = partitioner(0, 1, parameter)
        A, F, x = generate_problem_variable_step(f, x, BCs=BCs)
        U = np.linalg.solve(A, F)
        errors[-1]["errors"].append(norm(U, u, x))
        errors[-1]["Ms"].append(len(x))
        if DEBUG or DEBUG_C:
            print("  -Param: ", parameter, " : M:", len(x))

for error in errors:
    plt.loglog(error["Ms"], error["errors"], "-x", label=error["label"])
plt.legend()
plt.show()
