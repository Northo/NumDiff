import numpy as np
from functools import partial

### Problem Biharmonic ###
def biharmonic_u(x, y):
    return (
        (np.sin(np.pi*x) * np.sin(np.pi*y))**4
        *
        np.exp(-(x-0.5)**2 - (y-0.5)**2)
    )


@np.vectorize
def biharmonic_f(x, y):
    return (
        4*(
            6*np.pi**4*np.exp(x + y)*np.sin(np.pi*x)**4
            - 8*(
                4*np.pi*y**3*np.exp(x)*np.sin(np.pi*x)**4 - 6*np.pi*y**2*np.exp(x)*np.sin(np.pi*x)**4 - 6*np.pi**3*np.exp(x)*np.sin(np.pi*x)**2 + 4*(2*np.pi**2*x - np.pi**2)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 + (3*np.pi + 16*np.pi**3 - 2*np.pi*x**2 + 2*np.pi*x)*np.exp(x)*np.sin(np.pi*x)**4 + 4*(3*np.pi**3*np.exp(x)*np.sin(np.pi*x)**2 - 2*(2*np.pi**2*x - np.pi**2)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (np.pi + 8*np.pi**3 - np.pi*x**2 + np.pi*x)*np.exp(x)*np.sin(np.pi*x)**4)*y
            )*np.cos(np.pi*y)*np.exp(y)*np.sin(np.pi*y)**3
            + (
                4*y**4*np.exp(x)*np.sin(np.pi*x)**4 - 8*y**3*np.exp(x)*np.sin(np.pi*x)**4 - 8*(3*np.pi + 4*np.pi*x**3 + 16*np.pi**3 - 6*np.pi*x**2 - 4*(np.pi + 8*np.pi**3)*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 + (256*np.pi**4 + 4*x**4 - 8*(16*np.pi**2 + 1)*x**2 - 8*x**3 + 64*np.pi**2 + 4*(32*np.pi**2 + 3)*x + 1)*np.exp(x)*np.sin(np.pi*x)**4 + 6*np.pi**4*np.exp(x) - 24*(2*np.pi**3*x - np.pi**3)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x) - 12*(13*np.pi**4 - 6*np.pi**2*x**2 + 6*np.pi**2*x + 2*np.pi**2)*np.exp(x)*np.sin(np.pi*x)**2 + 8*(2*(np.pi - 2*np.pi*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (16*np.pi**2 - x**2 + x + 1)*np.exp(x)*np.sin(np.pi*x)**4 + 3*np.pi**2*np.exp(x)*np.sin(np.pi*x)**2)*y**2 - 4*(4*(np.pi - 2*np.pi*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (32*np.pi**2 - 2*x**2 + 2*x + 3)*np.exp(x)*np.sin(np.pi*x)**4 + 6*np.pi**2*np.exp(x)*np.sin(np.pi*x)**2)*y
            )*np.exp(y)*np.sin(np.pi*y)**4
            - 24*(2*np.pi**3*y*np.exp(x)*np.sin(np.pi*x)**4 - np.pi**3*np.exp(x)*np.sin(np.pi*x)**4)*np.cos(np.pi*y)*np.exp(y)*np.sin(np.pi*y)
            + 12*(
                6*np.pi**2*y**2*np.exp(x)*np.sin(np.pi*x)**4
                - 6*np.pi**2*y*np.exp(x)*np.sin(np.pi*x)**4
                + 6*np.pi**4*np.exp(x)*np.sin(np.pi*x)**2
                - 4*(2*np.pi**3*x - np.pi**3)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3
                - (13*np.pi**4 - 2*np.pi**2*x**2 + 2*np.pi**2*x + 2*np.pi**2)*np.exp(x)*np.sin(np.pi*x)**4
            )*np.exp(y)*np.sin(np.pi*y)**2
        )*np.exp(-x**2 - y**2 - 1/2)
    )


def get_biharmonic():
    return biharmonic_u, biharmonic_f


### Poisson problem exponential ###
def poisson_exp_u(x, y):
    return (
        (np.sin(np.pi*x) * np.sin(np.pi*y))**4
        *
        np.exp(-(x-0.5)**2 - (y-0.5)**2)
    )

def poisson_exp_f(x, y):
    return (
        2*(6*np.pi**2*np.exp(x + y)*np.sin(np.pi*x)**4*np.sin(np.pi*y)**2 - 4*(2*np.pi*y*np.exp(x)*np.sin(np.pi*x)**4 - np.pi*np.exp(x)*np.sin(np.pi*x)**4)*np.cos(np.pi*y)*np.exp(y)*np.sin(np.pi*y)**3 + (2*y**2*np.exp(x)*np.sin(np.pi*x)**4 + 4*(np.pi - 2*np.pi*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (16*np.pi**2 - 2*x**2 + 2*x + 1)*np.exp(x)*np.sin(np.pi*x)**4 - 2*y*np.exp(x)*np.sin(np.pi*x)**4 + 6*np.pi**2*np.exp(x)*np.sin(np.pi*x)**2)*np.exp(y)*np.sin(np.pi*y)**4)*np.exp(-x**2 - y**2 - 1/2)
    )


def get_poisson_exp_problem():
    return poisson_exp_u, poisson_exp_f


### Simple Poisson test problem ###
def f_term(x, y, k=1, l=1):
    """Manufactured solution"""
    return -np.sin(x * k * np.pi) * np.sin(y * l * np.pi)


def u_term(x, y, k=1, l=1):
    return f_term(x, y, k=k, l=l) / (-np.pi**2 * (k**2 + l**2))


def f_poisson_simple(x, y, k=[1], l=[1]):
    sum = 0
    for k in k:
        for l in l:
            sum += f_term(x, y, k, l)
    return sum


def u_poisson_simple(x, y, k=[1], l=[1]):
    sum = 0
    for k in k:
        for l in l:
            sum += u_term(x, y, k, l)
    return sum


def get_f_poisson_simple(k, l):
    return np.vectorize(
        partial(f_poisson_simple, k=k, l=l)
    )


def get_u_poisson_simple(k, l):
    return np.vectorize(
        partial(u_poisson_simple, k=k, l=l)
    )


def get_poisson_sin_problem(k=[1], l=[1]):
    return get_u_poisson_simple(k, l), get_f_poisson_simple(k, l)

