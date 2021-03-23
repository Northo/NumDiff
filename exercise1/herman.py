#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.integrate

class BoundaryCondition:
    DIRICHLET = 1
    NEUMANN = 2

    def __init__(self, type, value):
        self.type = type
        self.value = value

def f1(x):
    return np.cos(2*np.pi*x) + x

def anal(x, bc1, bc2):
    c1 = 0
    c2 = 0

    if bc1.type == BoundaryCondition.DIRICHLET:
        c1 = 1/(4*np.pi**2) + bc1.value
    elif bc1.type == BoundaryCondition.NEUMANN:
        c2 = bc1.value
    else:
        raise("Unknown boundary condition type")

    if bc2.type == BoundaryCondition.DIRICHLET:
        if bc1.type == BoundaryCondition.DIRICHLET:
            c2 = -c1 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc2.value
        elif bc1.type == BoundaryCondition.NEUMANN:
            c1 = -c2 - 1/12*(2*np.pi**2-3)/np.pi**2 + bc2.value
    elif bc2.type == BoundaryCondition.NEUMANN:
        if bc1.type == BoundaryCondition.NEUMANN:
            print("Warning: non-unique solution, imposing extra constraint u(0) == 0")
            c1 = 1/(4*np.pi**2) # enforces u(0) == 0
        c2 = -1/2 + bc2.value
    else:
        raise("Unknown boundary condition type")

    return c2*x + c1 + 1/12*(2*np.pi**2*x**3 - 3*np.cos(2*np.pi*x))/np.pi**2

def num(x, bc1, bc2, f=f1):
    hs = x[1:] - x[:-1]
    h_min = np.min(hs)
    h_max = np.max(hs)
    step_is_uniform = h_max - h_min < 1e-10
    h = hs[0] if step_is_uniform else 0 # hopefully crash by diving by zero if accessed where uniform step is assumed

    M = np.size(x)
    A = np.zeros((M,M))
    b = np.array([f(x) for x in x])

    print(f"Solving: M={M}, step_is_uniform={step_is_uniform}")

    # Set up equations for internal points
    # Approximation is 2nd order when the step size is uniform
    # Approximation is 1st order when the step size is non-uniform
    for m in range(1, M-1):
        # Account for variable step size (see Owren page 69 with a=1, c=0)
        # WARNING: is second order only if left- and right step sizes are equal
        h1 = x[m] - x[m-1]
        h2 = x[m+1] - x[m]
        A[m,m-1] = +2/(h1*(h1+h2))                  # reduces to +1/h**2 when h1 = h2 = h
        A[m,m+0] = -2/(h1*(h1+h2)) - 2/(h2*(h1+h2)) # reduces to -2/h**2 when h1 = h2 = h
        A[m,m+1] = +2/(h2*(h1+h2))                  # reduces to +1/h**2 when h1 = h2 = h

    # Set up equation for left boundary condition
    if bc1.type == BoundaryCondition.DIRICHLET:
        # Trivial equation 1 * U[0] = bc1.value
        A[0,0] = 1
        b[0] = bc1.value
    elif bc1.type == BoundaryCondition.NEUMANN and step_is_uniform:
        # 2nd order approximation (only supported for uniform step size)
        A[0,0] = -3/(2*h)
        A[0,1] = +2/h
        A[0,2] = -1/(2*h)
        b[0] = bc1.value
    else:
        raise("Unsupported boundary condition")

    # Set up equation for right boundary condition
    if bc2.type == BoundaryCondition.DIRICHLET:
        # Trivial equation 1 * U[M-1] = bc2.value
        A[M-1,M-1] = 1
        b[-1] = bc2.value
    elif bc2.type == BoundaryCondition.NEUMANN and step_is_uniform:
        # 2nd order approximation (only supported for uniform step size)
        # only for uniform step size
        A[M-1,M-3] = +1/(2*h)
        A[M-1,M-2] = -2/h
        A[M-1,M-1] = +3/(2*h)
        b[-1] = bc2.value
        if bc1.type == BoundaryCondition.NEUMANN:
            print("Warning: non-unique solution, imposing extra constraint u(0) == 0")
            A[:,0] = 0 # makes matrix singular, so must ultimately solve AU = b with least squares method
    else:
        raise("Unsupported boundary condition type")

    if bc1.type == BoundaryCondition.NEUMANN and bc2.type == BoundaryCondition.NEUMANN:
        U = np.linalg.lstsq(A, b)[0]
    else:
        U = np.linalg.solve(A, b)
    return U

def plot_solution(x, u, U, f, grid=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("u(x)")
    ax1.plot(x, u, label="analytic", linewidth=5, color="black")
    ax1.plot(x, U, label="numerical", linewidth=2, color="red")
    if grid:
        ax1.plot(x, U, "bo")
    ax1.legend()

    xfine = np.linspace(x[0], x[-1], 500)
    ax2.set_title("f(x)")
    ax2.plot(xfine, f(xfine), "k-", label="source")
    if grid:
        ax2.plot(x, f(x), "bo", label="source")

    plt.show()

def compare_num_anal(bc1, bc2, M=250, showplot=True, outpath=""):
    x = np.linspace(0, 1, M)
    U = num(x, bc1, bc2)
    u = anal(x, bc1, bc2)

    if outpath != "":
        table = np.column_stack((x, U, u))
        np.savetxt(outpath, table, header="x U u", comments="")

    if showplot:
        plot_solution(x, u, U, f1)

def errors(x, bc1, bc2):
    return err_disc, err_cont

def convergence_plot(bc1, bc2, showplot=False, outpath=""):
    Ms = [8,16,32,64,128,256,512,1024]
    hs = []
    errs_disc = []
    errs_cont = []
    for M in Ms:
        x, h = np.linspace(0, 1, M, retstep=True)
        hs.append(h)
        u = anal(x, bc1, bc2)
        U = num(x, bc1, bc2)
        #compare(x, u, U)
        err_disc = l2_disc(u-U) / l2_disc(u)
        err_cont = l2_cont(u-U, x) / l2_cont(u, x)
        errs_disc.append(err_disc)
        errs_cont.append(err_cont)

    if showplot:
        plt.loglog(Ms, errs_disc, label="discrete")
        plt.loglog(Ms, errs_cont, label="continuous")
        plt.show()

    if outpath != "":
        table = np.column_stack((hs, errs_disc, errs_cont))
        np.savetxt(outpath, table, header="h disc cont", comments="")

def l2_cont(y, x):
    # interpolate integration with trapezoid rule
    # TODO: interpolate with higher accuracy?
    return np.sqrt(np.trapz(y**2, x))

def l2_disc(y):
    N = np.size(y)
    return np.sqrt(1/N*np.sum(y**2))

def subdivide_interval(x1, x3, should_subdivide, tol):
    x = []
    def subdivide(x1, x3):
        x2 = (x1 + x3) / 2
        if should_subdivide(x1, x2, tol):
            subdivide(x1, x2)
        else:
            x.append(x1)

        if should_subdivide(x2, x3, tol):
            subdivide(x2, x3)
        else:
            x.append(x2)
    subdivide(x1, x3)
    x.append(x3) # append right point (only point missing)
    return np.array(x)

def manufactured_solution_mesh_refinement(maxM=1000):
    # symbolic manufactured solution
    # eps = 1e-3
    eps = 1e-1
    xsym = sympy.symbols("x")
    usym = sympy.exp(-1/eps*(xsym-1/2)**2)
    fsym = sympy.diff(sympy.diff(usym, xsym), xsym)
    absfsym = sympy.Abs(fsym)
    fsqsym = fsym**2

    # analytical solution
    ufunc = sympy.lambdify(xsym, usym, "numpy")
    ffunc = sympy.lambdify(xsym, fsym, "numpy")
    absffunc = sympy.lambdify(xsym, absfsym, "numpy")
    fsqfunc = sympy.lambdify(xsym, fsqsym, "numpy")

    bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, ufunc(0))
    bc2 = BoundaryCondition(BoundaryCondition.DIRICHLET, ufunc(1))

    def find_interval(x, x0):
        for m in range(0, len(x)-1):
            if x0 >= x[m] and x0 <= x[m+1]:
                return m
        raise f"Did not find interval for x0={x0}"

    def find_interval_with_maximum(x, quantity_function):
        i1_max = 0
        quantity_max = 0
        for i1 in range(0, len(x)-1):
            i2 = i1 + 1
            x1, x2 = x[i1], x[i2]
            quantity = quantity_function(x1, x2)
            if quantity > quantity_max:
                quantity_max = quantity
                i1_max = i1
        i2_max = i1_max + 1
        return i1_max, i2_max

    def insert_point(x, x0):
        m = find_interval(x, x0)
        x.insert(m+1, x0)
        return x

    def pointadder_umr(x):
        M = len(x) + 1 # increase number of points by one
        return np.linspace(0, 1, M)

    def pointadder_source(x):
        def total_source(x1, x2):
            return scipy.integrate.quad(absffunc, x1, x2)[0]
        i1, i2 = find_interval_with_maximum(x, total_source)
        x.insert(i2, (x[i1] + x[i2]) / 2) # split most critical interval
        return x

    def pointadder_truncerr(x):
        def truncerr(x1, x2):
            x1, x2, x3 = x1, (x1 + x2) / 2, x2 # swap, so x1 < x2 < x3
            h1, h2 = x2 - x1, x3 - x2
            u1, u2, u3 = ufunc(x1), ufunc(x2), ufunc(x3)
            f2 = ffunc(x2)
            return np.abs(2/(h1+h2) * ((u3-u2) / h2 - (u2-u1) / h1) - f2)
        i1, i2 = find_interval_with_maximum(x, truncerr)
        x.insert(i2, (x[i1] + x[i2]) / 2) # split most critical interval
        return x

    def pointadder_error(x):
        # solve numerically
        xx = np.array(x)
        U = num(xx, bc1, bc2, f=ffunc)

        # find interval with highest error, use find_interval_with_maximum(x, quantity_function):
        def absdiff(x):
            return np.abs(np.interp(x, xx, U) - ufunc(x))
        def error(x1, x2):
            return scipy.integrate.quad(absdiff, x1, x2)[0]
        i1, i2 = find_interval_with_maximum(x, error)

        # split interval
        x0 = (x[i1] + x[i2]) / 2
        x = insert_point(x, x0)
        if not np.abs(x0 - 0.5) < 1e-8:
            x = insert_point(x, 1-x0) # preserve symmetry unless inserting at x=0.5
        return x

    strategies = [
        {"label": "UMR", "pointadder": pointadder_umr},
        {"label": "AMR-error", "pointadder": pointadder_error},
        {"label": "AMR-source", "pointadder": pointadder_source},
        {"label": "AMR-truncerr", "pointadder": pointadder_truncerr},
    ]

    for strategy in strategies:
        strategy["npoints"] = []
        strategy["errs_disc"] = []
        strategy["errs_cont"] = []
        pointadder = strategy["pointadder"]
        label = strategy["label"]

        x = list(np.linspace(0, 1, 2))
        while len(x) < 100:
            x = pointadder(x)
            xx = np.array(x)
            u = ufunc(xx) # analytic solution
            U = num(xx, bc1, bc2, f=ffunc) # numerical solution
            err_disc = l2_disc(u-U) / l2_disc(u)
            err_cont = l2_cont(u-U, xx) / l2_cont(u, xx)
            strategy["npoints"].append(len(x))
            strategy["errs_disc"].append(err_disc)
            strategy["errs_cont"].append(err_cont)
            print(len(x), err_disc)

            # Uncomment to plot while you go
            # if len(x) == 27 and label == "AMR-source-f":
                # plot_solution(xx, u, U, ffunc, grid=True)
            # if strategy["label"] == "AMR-source-balance":
                # plot_solution(xx, u, U, ffunc, grid=True)
            # if strategy["label"] == "AMR-error":
                # plot_solution(xx, u, U, ffunc, grid=True)
            # if strategy["label"] == "AMR-source-absf":
                # plot_solution(xx, u, U, ffunc, grid=True)


    for i, strategy in enumerate(strategies):
        plt.loglog(strategy["npoints"], strategy["errs_disc"], marker="o", markersize=2, label=strategy["label"]+" (disc)", color=f"C{i}", linestyle="dashed")
        plt.loglog(strategy["npoints"], strategy["errs_cont"], marker="o", markersize=2, label=strategy["label"]+" (cont)", color=f"C{i}", linestyle="solid")
    plt.legend()
    plt.show()

# Task 1a
bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
# compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_neu.dat")
# convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_neu_err.dat")

# Task 1b
bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, 1)
bc2 = BoundaryCondition(BoundaryCondition.DIRICHLET, 1)
# compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_dir.dat")
# convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_dir_err.dat")

# Task 1c
bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 1/2)
# compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/neu_neu.dat")
# convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/neu_neu_err.dat")

manufactured_solution_mesh_refinement()
