#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy

def integrate(f, x1, x2, points=None):
    if x1 == x2:
        return 0
    # TODO: use gaussian quadrature?
    return scipy.integrate.quad(f, x1, x2, points=points)[0]

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

    """
    def dphi(i, y):
        # TODO: just use analytical expression instead?
        phivals[i] = 1 # modify temporarily
        dphivals = (phivals[1:] - phivals[:-1]) / (x[1:] - x[:-1])
        result = dphivals[np.searchsorted(x, y, "right") - 1]
        phivals[i] = 0 # reset phivals for memory efficiency
        return result
    """

    # def dphi(y, i, i1, i2, x):
        

    for i in range(0, M+2):
        for j in range(i, M+2):
            integrand = lambda y: dphi(i, y) * dphi(j, y)
            # print("astart")
            #A[i,j] = integrate(integrand, x, points=[x[i], x[j]])
            # print(x[max(i,j)], x[-1])
            # A[i,j] += integrate(integrand, x[0], x[min(i,j)])
            # A[i,j] += integrate(integrand, x[min(i,j)], x[max(i,j)])
            # A[i,j] += integrate(integrand, x[max(i,j)], x[-1])
            # print("aend")
            # xnorm, weights = np.polynomial.legendre.leggauss(2)
            # xsamples = 

            if j == i:
                if i > 0:
                    A[i,j] += 1/(x[i]-x[i-1]) 
                if i < M+1:
                    A[i,j] += 1/(x[i+1]-x[i])
            elif j == i + 1:
                if i < M+1: # x2 to x3
                    A[i,j] += 1/(x[i]-x[i+1])
            A[j,i] = A[i,j]

    for i in range(0, M+2):
        integrand = lambda y: phi(i, y) * f(y)

        # F[i] = integrate(integrand, x[0], x[-1])

        x1 = x[0]  if i == 0   else x[i-1]
        x2 = x[-1] if i == M+1 else x[i+1]
        # xint = np.linspace(x1, x2, 50)
        # F[i] = np.trapz(integrand(xint), xint)

        F[i] = scipy.integrate.quad(integrand, x1, x2, points=[x[i]])[0]


    F = F - A @ np.concatenate(([a], np.zeros(M), [b]))

    # Boundary conditions
    A = A[1:-1,1:-1]
    F = F[1:-1]

    U = np.linalg.solve(A, F)
    U = np.concatenate(([a], U, [b])) # restore boundary conditions
    return U

def adaptive(x, ffunc, maxM=20):
    xvar = sympy.var("x")
    u = sympy.integrate(-ffunc, xvar, xvar)
    u = sympy.lambdify(xvar, u, "numpy")
    a, b = u(x[0]), u(x[-1])

    f = sympy.lambdify(xvar, ffunc, "numpy")

    def error_measure(x1, x2, x, U, u):
        integrand = lambda y: (u(y) - np.interp(y, x, U))**2
        return scipy.integrate.quad(integrand, x1, x2)[0]

    while len(x) <= maxM:
        U = solve(x, f, a, b)

        errors = [error_measure(x[i], x[i+1], x, U, u) for i in range(0, len(x)-1)]
        intervalindex = np.argmax(errors)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.bar(x[:-1], errors, width=x[1:]-x[:-1], align="edge", edgecolor="black")
        ax2.plot(x, u(x), color="black", linewidth=5)
        ax2.plot(x, U, marker="o", color="red", linewidth=1, markersize=3)
        plt.show()

        x = np.insert(x, intervalindex + 1, (x[intervalindex]+x[intervalindex+1])/2)

"""
f = lambda x: -2
x = np.linspace(0, 1, 10)
# x = np.array([0, 0.1, 0.3, 0.85, 1])
a, b = 1, -2
U = solve(x, f, a, b)
plt.plot(x, -1/6*x**3 + (b-a+1/6)*x + a, linewidth=5, color="black")
plt.plot(x, U, linewidth=1, color="red")
plt.show()
"""

xvar = sympy.var("x")
# ffunc = (40000*xvar**2 - 200) * sympy.exp(-100*xvar**2)
# x = np.linspace(-1, 1, 2)
ffunc = -2
x = np.linspace(0, 1, 2)
adaptive(x, ffunc)
