#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def integrate(f, x, points=None):
    # TODO: use gaussian quadrature?
    return scipy.integrate.quad(f, x[0], x[-1], points=points)[0]

def solve(x, f, a, b):
    M = len(x) - 2 # [0, 1, ..., M, M+1]
    A = np.zeros((M+2, M+2)) # will later chop off ends
    F = np.zeros(M+2) # will later chop off ends

    phivals = np.zeros(len(x))

    def phi(i, y):
        # TODO: just use analytical expression instead?
        phivals[i] = 1 # modify temporarily
        result = np.interp(y, x, phivals)
        phivals[i] = 0 # reset phivals for memory efficiency
        return result

    def dphi(i, y):
        # TODO: just use analytical expression instead?
        phivals[i] = 1 # modify temporarily
        dphivals = (phivals[1:] - phivals[:-1]) / (x[1:] - x[:-1])
        result = dphivals[np.searchsorted(x, y, "right") - 1]
        phivals[i] = 0 # reset phivals for memory efficiency
        return result

    for i in range(0, M+2):
        for j in range(0, M+2):
            integrand = lambda y: dphi(i, y) * dphi(j, y)
            A[i,j] = integrate(integrand, x, points=[x[i], x[j]])

    for i in range(0, M+2):
        integrand = lambda y: phi(i, y) * f(y)
        F[i] = integrate(integrand, x)

    F = F - A @ np.concatenate(([a], np.zeros(M), [b]))

    # Boundary conditions
    A = A[1:-1,1:-1]
    F = F[1:-1]

    U = np.linalg.solve(A, F)
    U = np.concatenate(([a], U, [b])) # restore boundary conditions

    return U


f = lambda x: x
x = np.linspace(0, 1, 10)
# x = np.array([0, 0.1, 0.3, 0.85, 1])
a, b = 1, -2
U = solve(x, f, a, b)
plt.plot(x, -1/6*x**3 + (b-a+1/6)*x + a, linewidth=5, color="black")
plt.plot(x, U, linewidth=1, color="red")
plt.show()
