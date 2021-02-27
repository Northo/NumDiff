#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.optimize

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
    print("Step is uniform:", step_is_uniform)
    if step_is_uniform:
        h = hs[0]
    else:
        h = 0 # hopefully crash by diving by zero if accessed where uniform step is assumed

    M = np.size(x)
    A = np.zeros((M,M))
    b = np.array([f(x) for x in x])

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

def plot_num_over_anal(x, u, U):
    plt.plot(x, u, label="analytic", linewidth=5, color="black")
    plt.plot(x, U, label="numerical", linewidth=2, color="red")
    plt.legend()
    plt.show()

def compare_num_anal(bc1, bc2, M=500, showplot=True, outpath=""):
    x = np.linspace(0, 1, M)
    U = num(x, bc1, bc2)
    u = anal(x, bc1, bc2)

    if outpath != "":
        table = np.column_stack((x, U, u))
        np.savetxt(outpath, table, header="x U u", comments="")

    if showplot:
        plot_num_over_anal(x, u, U)

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
        plt.loglog(Ms, errs_disc[1], label="discrete")
        plt.loglog(Ms, errs_cont[1], label="continuous")
        plt.show()

    if outpath != "":
        table = np.column_stack((hs, errs_disc[1], errs_cont[1]))
        np.savetxt(outpath, table, header="h disc cont", comments="")

def l2_cont(y, x):
    # interpolate integration with trapezoid rule
    # TODO: interpolate with higher accuracy?
    return np.sqrt(np.trapz(y**2, x))

def l2_disc(y):
    N = np.size(y)
    return np.sqrt(1/N*np.sum(y**2))

def subdivide_interval(x1, x3, errfunc, tol):
    x = []
    def subdivide(x1, x3):
        x2 = (x1 + x3) / 2
        if np.abs(errfunc(x1, x2, x3)) < tol:
            x.append(x1) # error low enough, accept
        else:
            subdivide(x1, x2) # error too high, subdivide further
            subdivide(x2, x3)
    subdivide(x1, x3)
    x.append(x3) # append right point (only point missing)
    return np.array(x)

def plot_points(x, U):
    y = np.zeros(len(x))
    plt.plot(x, y, color="red", marker="|", markersize=20)
    plt.plot(x, U)
    plt.show()

def manufactured_solution_mesh_refinement(maxM=1000):
    # symbolic manufactured solution
    eps = 0.01
    xsym = sympy.symbols("x")
    usym = sympy.exp(-1/eps*(xsym-1/2)**2)
    fsym = sympy.diff(sympy.diff(usym, xsym), xsym)

    # analytical solution
    ufunc = sympy.lambdify(xsym, usym, "numpy")
    ffunc = sympy.lambdify(xsym, fsym, "numpy")

    def find_errs(x, order):
        u = ufunc(x) # analytic solution
        bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, ufunc(0))
        bc2 = BoundaryCondition(BoundaryCondition.DIRICHLET, ufunc(1))
        U = num(x, bc1, bc2, f=ffunc, order=order) # numerical solution
        err_disc = l2_disc(u-U) / l2_disc(u)
        err_cont = l2_cont(u-U, x) / l2_cont(u, x)
        # plot_points(x, U) # show solution and x-axis subdivision
        return err_disc, err_cont

    # Uniform Mesh Refinement (UMR) (order 1 and order 2)
    Ms_umr = [16, 32, 64, 128, 256, 512, 1024]
    errs_disc_umr_order1 = []
    errs_cont_umr_order1 = []
    errs_disc_umr_order2 = []
    errs_cont_umr_order2 = []
    for M in Ms_umr:
        x = np.linspace(0, 1, M) # divide space uniformly
        err_disc_order1, err_cont_order1 = find_errs(x, 1)
        err_disc_order2, err_cont_order2 = find_errs(x, 2)
        errs_disc_umr_order1.append(err_disc_order1)
        errs_cont_umr_order1.append(err_cont_order1)
        errs_disc_umr_order2.append(err_disc_order2)
        errs_cont_umr_order2.append(err_cont_order2)

    # analytical expression for local truncation error, if we use (+1,-2,+1)-stencil with variable step size
    def trunc_err(x1, x2, x3): 
        h1 = x2 - x1
        h2 = x3 - x2
        u1, u2, u3 = ufunc(x1), ufunc(x2), ufunc(x3)
        f2 = ffunc(x2)
        return 2/(h1+h2) * ((u3-u2) / h2 - (u2-u1) / h1) - f2

    # Adaptive Mesh Refinement (AMR) (order between 1 and 2)
    tol_amr = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
    Ms_amr = []
    errs_disc_amr = []
    errs_cont_amr = []
    for tol in tol_amr:
        x = subdivide_interval(0, 1, trunc_err, tol)
        M = len(x)
        Ms_amr.append(M)
        err_disc, err_cont = find_errs(x, 2)
        errs_disc_amr.append(err_disc)
        errs_cont_amr.append(err_cont)

    plt.loglog(Ms_umr, errs_disc_umr_order2, marker="x", label="UMR (order 2)")
    plt.loglog(Ms_umr, errs_cont_umr_order2, marker="x", label="UMR (order 2)")
    plt.loglog(Ms_umr, errs_disc_umr_order1, marker="x", label="UMR (order 1)")
    plt.loglog(Ms_umr, errs_cont_umr_order1, marker="x", label="UMR (order 1)")
    plt.loglog(Ms_amr, errs_disc_amr, marker="x", label="AMR (order \"1-2\")")
    plt.loglog(Ms_amr, errs_cont_amr, marker="x", label="AMR (order \"1-2\")")
    plt.legend()
    plt.show()

# Task 1a
bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_neu.dat")
convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_neu_err.dat")

# Task 1b
bc1 = BoundaryCondition(BoundaryCondition.DIRICHLET, 1)
bc2 = BoundaryCondition(BoundaryCondition.DIRICHLET, 1)
compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_dir.dat")
convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/dir_dir_err.dat")

# Task 1c
bc1 = BoundaryCondition(BoundaryCondition.NEUMANN, 0)
bc2 = BoundaryCondition(BoundaryCondition.NEUMANN, 1/2)
compare_num_anal(bc1, bc2, showplot=True, outpath="../report/exercise1/neu_neu.dat")
convergence_plot(bc1, bc2, showplot=True, outpath="../report/exercise1/neu_neu_err.dat")

# manufactured_solution_mesh_refinement()
