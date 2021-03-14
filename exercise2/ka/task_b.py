#!/usr/bin/python3
from heateq import *
from functools import partial

PLOT_SAMPLES = True
OUT_DIR = "../../report/exercise2/data_ka/"


##################################
### Defining equation - IC/BCs ###
##################################


def u0(x):
    """ Initial condition """
    # return 10*np.sin(x*np.pi)
    return np.sin(x * np.pi)


# Boundary conditions (Dirchlet)
bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)


def analytical(x, t):
    """ Analytical solution of manufactured dirchlet problem """
    return np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)


##########################
### Numerical solution ###
##########################


# All error functions must have the call sign.
# error_func(U:ndarray, u:callable, x:ndarray)
ERROR_FUNCTIONS = {
    "L2 discrete": lambda U, u, x: l2_discrete_relative_error(u(x), U),
    "L2 continous interpolation": lambda U, u, x: L2_continous_relative_error(u, continous_continuation(x, U)),
    "L2 continous step": lambda U, u, x: L2_continous_relative_error(u, piecewise_constant_continuation(x, U)),
}

## Partitioners ##
AMR_simple = partial(
    partition_interval,
    error_function=curvature_estimator(u0)
)
AMR_min_dist = partial(
    partition_interval,
    error_function=min_dist_mixin(AMR_simple, 0.1)
)

M = np.arange(10, 1000, 100)  # Used in UMR

# Find errors for each methdo defined in methods
methods = [
    # partitioner, parameters, error_function, solver, label
    [np.linspace, M, ERROR_FUNCTIONS["L2 continous interpolation"], backward_euler, "UMR contionus BE"],
    [np.linspace, M, ERROR_FUNCTIONS["L2 continous interpolation"], crank_nicolson, "UMR contionus CN"],
    [np.linspace, M, ERROR_FUNCTIONS["L2 discrete"], backward_euler, "UMR discrete BE"],
    [AMR_simple, np.geomspace(0.0000046, 0.1, 10), ERROR_FUNCTIONS["L2 continous step"], backward_euler, "AMR continous BE"],
    [AMR_simple, np.geomspace(0.0000046, 0.1, 10), ERROR_FUNCTIONS["L2 continous step"], crank_nicolson, "AMR continous CN"],
]

N = 100  # Number of time steps
errors = []
for partitioner, parameters, error_function, method, label in methods:
    Ms, errors_for_Ms = find_error(
        partitioner,
        parameters,
        error_function,
        analytical_func=analytical,
        method=method,
        bc1=bc1, bc2=bc2,
        u0=u0,
        N=N,
        t_end=1,
    )
    errors.append({"errors": errors_for_Ms, "Ms": Ms, "label": label})

for error in errors:
    plt.plot(error["Ms"], error["errors"], label=error["label"])
plt.legend()
plt.show()
