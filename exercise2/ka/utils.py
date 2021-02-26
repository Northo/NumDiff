# Utils
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt


# Hermdog's BC class from exercise 1
class BoundaryCondition:
    DIRCHLET = 1
    NEUMANN = 2

    def __init__(self, type, value):
        self.type = type
        if callable(value):
            self.value = value
        else:
            self.value = lambda t: value


class Grid:

    def __init__(self, is_uniform, x):
        self.is_uniform = is_uniform
        self.x = x
        if is_uniform:
            self.h = x[1] - x[0]
        else:
            self.h = 0


def discrete_l2_norm(V):
    """ discrete l2 norm """
    return np.linalg.norm(V) / np.sqrt(len(V))


def l2_discrete_relative_error(U_ref, U):
    """ Compute and return the l2 discrete relative error """

    return discrete_l2_norm(U_ref - U) / discrete_l2_norm(U_ref)


def L2_continous_norm(v, x_min=0, x_max=1):
    """ Compute and return the L2 continous norm """
    return np.sqrt(quad(lambda x: v(x) ** 2, x_min, x_max)[0])


def L2_continous_relative_error(U_ref, U):
    """ Compute and return the L2 continous relative error """
    return L2_continous_norm(lambda x: U_ref(x) - U(x)) / L2_continous_norm(U_ref)


def piecewise_constant_continuation(xr, ur):
    """
    make a piecewise constant function of spacial coordinate x from a reference solution u

    Parameters:
        xr : x grid for the reference solution
        ur : Array, the reference solution
    Returns:
        numpy.piecewise function, piecewise constant funciton of x
    """

    return lambda x: np.piecewise(
        x,
        [xr[i] <= x < xr[j] for (i, j) in zip(range(len(ur) - 1), range(1, len(ur)))],
        ur,
    )


def continous_continuation(xr, ur):
    """ Cont. continuation using interpolation """

    return lambda x: np.interp(x, xr, ur)



