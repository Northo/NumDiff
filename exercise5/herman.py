#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
import sympy

def write_table_to_file(path, table, headers):
    np.savetxt(path, table, header=" ".join(headers), comments="")
    print(f"Wrote to {path}")

def write_columns_to_file(path, columns, headers):
    max_length = np.max([len(column) for column in columns])
    for i in range(0, len(columns)):
        length = len(columns[i])
        column = np.full(max_length, np.nan)
        column[0:length] = columns[i]
        columns[i] = column
    write_table_to_file(path, np.transpose(columns), headers)

class Problem:
    def __init__(self, f, x1x2, u1u2, label=""):
        self.label = label

        self.x1, self.x2 = x1x2
        self.u1, self.u2 = u1u2

        # solve analytically
        xvar = sympy.var("x")
        up = sympy.integrate(-f, xvar, xvar) # particular solution

        # general solution u = A + Bx + up
        A = (self.u1-up.subs(xvar,self.x1)+self.u2-up.subs(xvar,self.x2))/2
        B = (self.u1-up.subs(xvar,self.x1)-self.u2+up.subs(xvar,self.x2))/(2*self.x1) if self.x1 != 0 else (self.u2-up.subs(xvar,self.x2)-self.u1+up.subs(xvar,self.x1))/(2*self.x2)
        u = A + B*xvar + up # general solution
        print(u)

        # make callable and efficient
        self.u = sympy.lambdify(xvar, u, "numpy")
        self.f = sympy.lambdify(xvar, f, "numpy")

        self.strategy = None

    def solve_uniform(self, M):
        # TODO: make M vs M + 2 consistent
        x = np.linspace(self.x1, self.x2, M+2)
        self.U = self.solve(x)

    def plot_analytic(self):
        x = np.linspace(self.x1, self.x2, 500)
        plt.plot(x, self.u(x), color="black", linewidth=5, label="analytic", alpha=0.25)

    def plot_numeric(self):
        plt.plot(self.x, self.U, marker=None, color="red", linewidth=1, markersize=3, label="numerical")

    def plot(self, show=False):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        widths = self.x[1:] - self.x[:-1]
        ax1.bar(self.x[:-1], self.errors, width=widths, align="edge", edgecolor="black")

        x = np.linspace(self.x1, self.x2, 500)
        ax2.plot(x, self.u(x), color="black", linewidth=5, label="analytic")
        ax2.plot(self.x, self.U, marker="o", color="red", linewidth=1, markersize=3, label="numerical")
        ax2.legend()

        if show:
            plt.show()

    def error_measure(self, i1=0, i2=None):
        if i2 == None:
            i2 = len(self.x) - 1 # by default give error on whole domain
        integrand = lambda y: (self.u(y) - np.interp(y, self.x, self.U))**2
        return np.sqrt(sum(scipy.integrate.quad(integrand, self.x[i], self.x[i+1])[0] for i in range(i1, i2)))

    def solve(self, x):
        M = len(x) - 2 # [0, 1, ..., M, M+1]
        A = np.zeros((M+2, M+2)) # will later chop off ends
        F = np.zeros(M+2) # will later chop off ends

        print(f"Solving M = {M}")

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
                F[i] += scipy.integrate.fixed_quad(integrand, x[i-1], x[i], n=20)[0]
            if i < M+1:
                integrand = lambda y: (x[i+1]-y)/(x[i+1]-x[i]) * self.f(y)
                F[i] += scipy.integrate.fixed_quad(integrand, x[i], x[i+1], n=20)[0]

        # Boundary conditions
        F = F - A @ np.concatenate(([self.u1], np.zeros(M), [self.u2]))
        A = A[1:-1,1:-1]
        F = F[1:-1]

        A = scipy.sparse.csr_matrix(A)
        U = scipy.sparse.linalg.spsolve(A, F)
        U = np.concatenate(([self.u1], U, [self.u2])) # restore boundary conditions
        self.x = x
        self.U = U

        # Collect errors
        self.errors = [self.error_measure(i, i+1) for i in range(0, len(self.x)-1)]
        self.errors = np.array(self.errors)

        if self.strategy == "avgerror":
            self.referror = 0.99 * np.mean(self.errors)
        elif self.strategy == "maxerror":
            self.referror = 0.7 * np.max(self.errors)

        return U

    def refine(self, M0, refiner, callback, plot, write, label, steps=None, maxM=None):
        assert not (steps is None and maxM is None), "Must provide either steps or maxM"
        x = np.linspace(self.x1, self.x2, M0) # initial uniform grid
        step = 0
        while True:
            if steps is not None and step >= steps:
                break
            elif maxM is not None and len(x) > maxM:
                break
            self.solve(x)
            if callback:
                callback(step)
            if write:
                self.write(label, step)
            if plot:
                self.plot(show=True)
            x = refiner(x)
            step += 1

    def refine_uniformly(self, callback=None, M0=2, steps=12, plot=False, write=False):
        def refiner(x):
            return np.linspace(self.x1, self.x2, 2*(len(x)-1)+1)
        self.refine(M0, refiner, callback, plot, write, "UMR", steps=steps)

    def refine_adaptively(self, strategy, callback=None, M0=2, steps=None, maxM=None, plot=False, write=False):
        self.strategy = strategy

        def refiner(x):
            newx = np.array([])
            for i in range(0, len(x)-1):
                newx = np.append(newx, x[i])
                if self.errors[i] >= self.referror:
                    newx = np.append(newx, (x[i] + x[i+1]) / 2)
            newx = np.append(newx, x[-1])
            return newx
        
        self.refine(M0, refiner, callback, plot, write, "AMR", steps=steps, maxM=maxM)

    def convergence_plot(self, plot=False, write=False, M0=20, maxM=2050):
        i = 0
        Ms, Es = np.full((3, 50), np.nan), np.full((3, 50), np.nan) # allocate enough space for all iterations

        def callback(step):
            Ms[i,step] = len(self.x) - 2
            Es[i,step] = self.error_measure()

        self.refine_uniformly(callback, steps=9, M0=9)
        plt.loglog(Ms[i], Es[i], marker="o", label="uniform")

        i = 1
        self.refine_adaptively("avgerror", callback, M0=M0, maxM=maxM) # TODO: start at 2 or 20?
        plt.loglog(Ms[i], Es[i], marker="o", label="avgerror")

        i = 2
        self.refine_adaptively("maxerror", callback, M0=M0, maxM=maxM) # TODO: start at 2 or 20?
        plt.loglog(Ms[i], Es[i], marker="o", label="maxerror")

        if write:
            columns = [Ms[0], Es[0], Ms[1], Es[1], Ms[2], Es[2]]
            headers = [ "M1",  "E1",  "M2",  "E2",  "M3",  "E3"]
            write_columns_to_file(f"../report/exercise5/data/convergence/{self.label}-convergence.dat", columns, headers)

        if plot:
            plt.legend()
            plt.show()

    def write(self, dir, step, normalize_errors=True):
        M = len(self.x)

        errors = self.errors / np.max(self.errors) if normalize_errors else self.errors
        if self.strategy is not None and normalize_errors:
            referror = self.referror / np.max(self.errors)
        elif self.strategy is not None and not normalize_errors:
            referror = self.referror
        else:
            referror = -1 # dummy, hide from plots

        path = f"../report/exercise5/data/{dir}/{self.label}-step{step}.dat"
        columns = [self.x, self.U, np.concatenate((errors, [0])), np.full(len(self.x), referror)]
        headers = [   "x",    "U",                           "E",   "refE"]
        write_columns_to_file(path, columns, headers)

x = sympy.var("x")

#label, f, (x1, x2), (u1, u2) = "f1", -2, (0, 1), (0, 1)
#label, f, (x1, x2), (u1, u2) = "f2", (40000*x**2 - 200) * sympy.exp(-100*x**2), (-1, +1), (np.exp(-100), np.exp(-100))
#label, f, (x1, x2), (u1, u2) = "f3", (4000000*x**2 - 2000) * sympy.exp(-1000*x**2), (-1, +1), (np.exp(-1000), np.exp(-1000))
#label, f, (x1, x2), (u1, u2) = "f4", 2/9*x**(-4/3), (0, 1), (0, 1) # TODO: how to deal with singularity at f(0)?

params = [
    ("f1", -2, (0, 1), (0, 1)),
    ("f2", -(40000*x**2 - 200) * sympy.exp(-100*x**2), (-1, +1), (np.exp(-100), np.exp(-100))),
    ("f3", -(4000000*x**2 - 2000) * sympy.exp(-1000*x**2), (-1, +1), (np.exp(-1000), np.exp(-1000))),
    ("f4", 2/9*x**(-4/3), (0, 1), (0, 1)), # TODO: how to deal with singularity at f(0)?
]

probs = [Problem(f, (x1, x2), (u1, u2), label=label) for label, f, (x1, x2), (u1, u2) in params]
for prob in probs:
    # prob.refine_uniformly(M0=8, steps=3, plot=False, write=True)
    # prob.refine_adaptively("avgerror", M0=20, steps=4, plot=False, write=True)
    prob.convergence_plot(write=True, maxM=2050)

# prob = Problem(f, (x1, x2), (u1, u2), label=label)
#prob.solve_adaptive(np.array([-1, -0.5, 0, 0.1, 0.3, 0.8, 1]))
#prob.refine_adaptively("avgerror", M0=20, steps=4, plot=False)
#prob.refine_uniformly()
#prob.solve_uniform(80)
#prob.plot(show=True)
