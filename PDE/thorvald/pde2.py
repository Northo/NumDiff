import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import dst, idst
import sympy
import problems
import time

########################
### Helper functions ###
########################
def dst2D(x, **kwargs):
    """Discrete sine transform in 2D.
    kwargs are passed to scipy.fft"""
    return dst(dst(x, axis=0, **kwargs), axis=1, **kwargs)


def idst2D(x, **kwargs):
    """Discrete sine transform in 2D.
    kwargs are passed to scipy.fft"""
    return idst(idst(x, axis=0, **kwargs), axis=1, **kwargs)


def TST(N, a=-2, b=1):
    """Tridiagonal, symmetric, toeplitz matrix"""
    return scipy.sparse.diags(
        [np.full(N - 1, b), np.full(N, a), np.full(N - 1, b)],
        [-1, 0, 1],
    )


def get_eigval_array(N, eigval_func):
    k = l = np.arange(1, N + 1)
    kk, ll = np.meshgrid(k, l)
    return eigval_func(N, kk, ll)


def get_mesh(N, reth=True, internal=True):
    """Generate the mesh for the problem with N internal points in one direction.
    reth: return the steplength h
    internal: only return internal points"""
    x = y = np.linspace(0, 1, N + 2)
    if internal:
        x = x[1:-1]
        y = y[1:-1]
    xx, yy = np.meshgrid(x, y)
    if reth:
        h = 1 / (N + 2 - 1)  # N + 2 points in total, N internal
        return xx, yy, h
    else:
        return xx, yy


def get_eigval(N, eigval_func):
    """Calculate the eigval array.
    eigval_func: callable(N, k, l)."""
    k = l = np.arange(1, N + 1)
    kk, ll = np.meshgrid(k, l)
    return eigval_func(N, kk, ll)


################################
### Stencils and eigenvalues ###
################################
def five_point_stencil(N, a=-4, b=1):
    """Generate a five point stencil"""
    K = TST(N, a=a / 2, b=b)
    I = scipy.sparse.eye(N)
    return scipy.sparse.kron(I, K) + scipy.sparse.kron(K, I)


def five_point_eigenval(N, k, l):
    """Eigenvalue of the five point stencil, for eigenvector k,l"""
    return 2 * (np.cos(k * np.pi / (N + 1)) + np.cos(l * np.pi / (N + 1))) - 4


def nine_point_stencil(N):
    """Generate a nine point stencil"""
    five_point = five_point_stencil(N, a=-10 / 3, b=2 / 3)
    SIGMA = scipy.sparse.diags([np.ones(N - 1), np.ones(N - 1)], [-1, 1])
    diagonal_points = scipy.sparse.kron(SIGMA, SIGMA) / 6

    return five_point + diagonal_points


def nine_point_eigenval(N, k, l):
    return (
        -10 / 3
        + 4 / 6 * (np.cos(k * np.pi / (N + 1)) * np.cos(l * np.pi / (N + 1)))
        + 4 / 3 * (np.cos(k * np.pi / (N + 1)) + np.cos(l * np.pi / (N + 1)))
    )


###############
### Solvers ###
###############
def fps(F, eigval_array, fps_kwargs={"type": 1, "norm": "ortho"}, **kwargs):
    """Fast poisson solver.
    eigval_array are the eigenvalues corresponding to the stencil to solve for
    kwargs are passed on to the dst transform."""
    fps_kwargs.update(kwargs)
    return idst2D(dst2D(F, **fps_kwargs) / eigval_array, **fps_kwargs)


class Solver:
    def __init__(
        self,
        f,
        N,
        anal_u=None,
        store_solution=False,
        err_order=2,
        err_relative=True,
    ):
        print(f"{self.__class__.__name__} for N = {N}")
        self.f = f
        self.N = N
        self.U = None
        self.store_solution = store_solution
        self.xx, self.yy, self.h = get_mesh(self.N, internal=False)
        self.F = f(self.xx, self.yy)
        if callable(anal_u):
            self.anal = anal_u(self.xx, self.yy)[1:-1, 1:-1]
        else:
            self.anal = anal_u

        # Generate errfunction
        def errfunc(U, u):
            U = U.flatten()
            u = u.flatten()
            if err_relative:
                return np.linalg.norm(U - u, ord=err_order) / np.linalg.norm(
                    u, ord=err_order
                )
            else:
                return np.linalg.norm(U - u, ord=err_order)

        self.errfunc = errfunc
        self.errors = {}
        self.times = {}

    def solve(self):
        print("\tSolving five point...")
        U5 = self.solve_five()
        if self.store_solution:
            self.U5 = U5
        self.errors.update({"five": self.errfunc(U5, self.anal)})
        del U5

        print("\tSolving nine point...")
        U9 = self.solve_nine()
        if self.store_solution:
            self.U9 = U9
        self.errors.update({"nine": self.errfunc(U9, self.anal)})


class PoissonSolver(Solver):
    """Solver for the Poisson equation, assuming u = 0 on border"""

    def solve_five(self):
        # We only want to solve for internal points,
        # as border is assumed zero.
        F_internal = self.F[1:-1, 1:-1]
        start_time = time.time()
        U = fps(self.h ** 2 * F_internal, get_eigval(self.N, five_point_eigenval))
        self.times.update({"five": time.time() - start_time})
        return U

    def solve_nine(self):
        F_stencil = five_point_stencil(self.N + 2, a=2 / 3, b=1 / 12)
        F = (F_stencil @ self.F.flatten()).reshape(self.N + 2, self.N + 2)
        F_internal = F[1:-1, 1:-1]
        start_time = time.time()
        U = fps(self.h ** 2 * F_internal, get_eigval(self.N, nine_point_eigenval))
        self.times.update({"nine": time.time() - start_time})
        return U


class BiharmonicSolver(Solver):
    """Solver for the Poisson equation, assuming nabla^2 u = u = 0 on border"""

    def solve_five(self):
        # We only want to solve for internal points,
        # as border is assumed zero.
        F_internal = self.F[1:-1, 1:-1]
        eigval = get_eigval(self.N, five_point_eigenval)
        start_time = time.time()
        G = fps(self.h ** 2 * F_internal, eigval)
        U = fps(self.h ** 2 * G, eigval)
        self.times.update({"five": time.time() - start_time})
        return U

    def solve_nine(self):
        eigval = get_eigval(self.N, nine_point_eigenval)
        F_stencil = five_point_stencil(self.N + 2, a=2 / 3, b=1 / 12)
        F = (F_stencil @ self.F.flatten()).reshape(self.N + 2, self.N + 2)
        F_internal = F[1:-1, 1:-1]
        G_stencil = five_point_stencil(self.N, a=2 / 3, b=1 / 12)
        start_time = time.time()
        G = fps(self.h ** 2 * F_internal, eigval)
        # TODO: better to use F, instead of F_stencil @ G
        G = (G_stencil @ G.flatten()).reshape(self.N, self.N)
        U = fps(self.h ** 2 * G, eigval)
        self.times.update({"nine": time.time() - start_time})
        return U


class Series:
    def __init__(self, u, f, solver, N_min, N_max, N_num, name="", store_solution_index=None):
        self.Ns = np.geomspace(N_min, N_max, N_num, dtype=int)
        self.hs = 1/(self.Ns + 2 - 1)
        self.u = u
        self.f = f
        self.errors = {"five": [], "nine": []}
        self.times = {"five": [], "nine": []}
        self.store_solution_index = store_solution_index % N_num
        self.solver = solver

    def run(self):
        for i, N in enumerate(self.Ns):
            store = (i == self.store_solution_index)
            solver = self.solver(f, N=N, anal_u=u, store_solution=store)
            solver.solve()
            if store:
                self.stored_solver = solver
            for error in solver.errors:
                self.errors[error].append(solver.errors[error])
            for time in solver.times:
                self.times[time].append(solver.times[time])

    def plot_error(self):
        for error in self.errors:
            plt.loglog(self.Ns, self.errors[error], label=error)
        plt.loglog(self.Ns, self.hs**2, linestyle="dashed", label="h2")
        plt.loglog(self.Ns, self.hs**4, linestyle="dashed", label="h4")
        plt.legend()
        plt.show()

    def plot_times(self):
        for time in self.times:
            plt.loglog(self.Ns, self.times[time], label=time)
        plt.legend()
        plt.show()

    def plot_solution(self):
        if not self.stored_solver:
            return
        plt.imshow(self.stored_solver.U9)
        plt.colorbar()
        plt.show()

    def write(self):
        print("wrote", self.write_error())
        print("wrote", self.write_time())
        if self.store_solution_index is not None:
            print("wrote", self.write_solution())

    def write_error(self):
        minmaxnum = f"{self.Ns[0]}:{self.Ns[-1]}:{len(self.Ns)}"
        filename = f"error_{self.solver.__name__}_N_{minmaxnum}.dat"
        data = np.array([self.Ns, self.errors["five"], self.errors["nine"]]).T
        header = "N five nine"
        np.savetxt(filename, data, header=header, comments="")
        return filename

    def write_time(self):
        minmaxnum = f"{self.Ns[0]}:{self.Ns[-1]}:{len(self.Ns)}"
        filename = f"time_{self.solver.__name__}_N_{minmaxnum}.dat"
        data = np.array([self.Ns, self.times["five"], self.times["nine"]]).T
        header = "N five nine"
        np.savetxt(filename, data, header=header, comments="")
        return filename

    def write_solution(self):
        solver = self.stored_solver
        xx, yy, U9 = solver.xx, solver.yy, solver.U9
        xx = xx[1:-1, 1:-1]
        yy = yy[1:-1, 1:-1]
        data = np.array([d.flatten() for d in [xx, yy, U9]]).T
        header = "x y U"
        N = self.Ns[self.store_solution_index]
        filename = f"solution_{solver.__class__.__name__}_N_{N}.dat"
        np.savetxt(filename, data, header=header, comments="")
        return filename

######################
### Let's do this! ###
######################

CASE = 2

if CASE == 0:
    u, f = problems.get_poisson_sin_problem(k=[1], l=[3, 4])
    solver = PoissonSolver
elif CASE == 1:
    u, f = problems.get_poisson_exp_problem()
    solver = PoissonSolver
elif CASE == 2:
    u, f = problems.get_biharmonic()
    solver = BiharmonicSolver
else:
    raise("Invalid case!")

s = Series(u, f, solver, *(20, 21, 1), name="", store_solution_index=0)
s.run()
s.plot_times()
# s.plot_error()
# s.plot_solution()
s.write()
