#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy

class Problem:
    def __init__(self, f, x1x2, u1u2):
        self.x1, self.x2 = x1x2
        self.u1, self.u2 = u1u2

        # solve analytically
        xvar = sympy.var("x")
        up = sympy.integrate(-f, xvar, xvar) # particular solution

        # general solution u = A + Bx + up
        A = (self.u1-up.subs(xvar,self.x1)+self.u2-up.subs(xvar,self.x2))/2
        B = (self.u1-up.subs(xvar,self.x1)-self.u2+up.subs(xvar,self.x2))/(2*self.x1) if self.x1 != 0 else (self.u2-up.subs(xvar,self.x2)-self.u1+up.subs(xvar,self.x1))/(2*self.x2)
        u = A + B*xvar + up # general solution

        # make callable and efficient
        self.u = sympy.lambdify(xvar, u, "numpy")
        self.f = sympy.lambdify(xvar, f, "numpy")

    def solve_uniform(self, M):
        # TODO: make M vs M + 2 consistent
        x = np.linspace(self.x1, self.x2, M+2)
        self.U = self.solve(x)

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        errors = self.get_errors()
        widths = self.x[1:] - self.x[:-1]
        ax1.bar(self.x[:-1], errors, width=widths, align="edge", edgecolor="black")

        x = np.linspace(self.x1, self.x2, 500)
        ax2.plot(x, self.u(x), color="black", linewidth=5, label="analytic")
        ax2.plot(self.x, self.U, marker="o", color="red", linewidth=1, markersize=3, label="numerical")
        ax2.legend()

        plt.show()

    def solve(self, x):
        M = len(x) - 2 # [0, 1, ..., M, M+1]
        A = np.zeros((M+2, M+2)) # will later chop off ends
        F = np.zeros(M+2) # will later chop off ends

        for i in range(0, M+2):
            for j in range(i, M+2):
                if j == i:
                    if i > 0:
                        A[i,j] += 1/(x[i]-x[i-1]) 
                    if i < M+1:
                        A[i,j] += 1/(x[i+1]-x[i])
                elif j == i + 1:
                    if i < M+1:
                        A[i,j] += 1/(x[i]-x[i+1])
                A[j,i] = A[i,j] # symmetric

        for i in range(0, M+2):
            if i > 0:
                integrand = lambda y: (y-x[i-1])/(x[i]-x[i-1]) * self.f(y)
                F[i] += scipy.integrate.quad(integrand, x[i-1], x[i])[0]
            if i < M+1:
                integrand = lambda y: (x[i+1]-y)/(x[i+1]-x[i]) * self.f(y)
                F[i] += scipy.integrate.quad(integrand, x[i], x[i+1])[0]

        # Boundary conditions
        F = F - A @ np.concatenate(([self.u1], np.zeros(M), [self.u2]))
        A = A[1:-1,1:-1]
        F = F[1:-1]

        U = np.linalg.solve(A, F)
        U = np.concatenate(([self.u1], U, [self.u2])) # restore boundary conditions
        self.x = x
        self.U = U
        return U

    def get_errors(self):
        def error_measure(x1, x2):
            integrand = lambda y: (self.u(y) - np.interp(y, self.x, self.U))**2
            return scipy.integrate.quad(integrand, x1, x2)[0]
        errors = [error_measure(self.x[i], self.x[i+1]) for i in range(0, len(self.x)-1)]
        return errors

    def refine_adaptively(self, maxM=20):
        x = np.linspace(self.x1, self.x2, 2)

        while len(x) <= maxM:
            self.solve(x)
            errors = self.get_errors()
            intervalindex = np.argmax(errors)

            self.plot()

            x = np.insert(x, intervalindex + 1, (x[intervalindex]+x[intervalindex+1])/2)

x = sympy.var("x")

# f, (x1, x2), (u1, u2) = -2, (0, 1), (0, 1)
# f, (x1, x2), (u1, u2) = (40000*x**2 - 200) * sympy.exp(-100*x**2), (-1, +1), (np.exp(-100), np.exp(-100))
# f, (x1, x2), (u1, u2) = (4000000*x**2 - 2000) * sympy.exp(-1000*x**2), (-1, +1), (np.exp(-1000), np.exp(-1000))
f, (x1, x2), (u1, u2) = 2/9*x**(-4/3), (0, 1), (0, 1) # TODO: how to deal with singularity at f(0)?

prob = Problem(f, (x1, x2), (u1, u2))
# prob.solve_uniform(80)
# prob.solve_adaptive(np.array([-1, -0.5, 0, 0.1, 0.3, 0.8, 1]))
prob.refine_adaptively()
prob.plot()
