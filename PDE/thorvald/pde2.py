import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import dst, idst
import sympy
import problems

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
        return fps(self.h ** 2 * F_internal, get_eigval(self.N, five_point_eigenval))

    def solve_nine(self):
        F_stencil = five_point_stencil(self.N + 2, a=2 / 3, b=1 / 12)
        F = (F_stencil @ self.F.flatten()).reshape(self.N + 2, self.N + 2)
        F_internal = F[1:-1, 1:-1]
        return fps(self.h ** 2 * F_internal, get_eigval(self.N, nine_point_eigenval))


class BiharmonicSolver(Solver):
    """Solver for the Poisson equation, assuming nabla^2 u = u = 0 on border"""

    def solve_five(self):
        # We only want to solve for internal points,
        # as border is assumed zero.
        F_internal = self.F[1:-1, 1:-1]
        eigval = get_eigval(self.N, five_point_eigenval)
        G = fps(self.h ** 2 * F_internal, eigval)
        U = fps(self.h ** 2 * G, eigval)
        return U

    def solve_nine(self):
        eigval = get_eigval(self.N, nine_point_eigenval)
        F_stencil = five_point_stencil(self.N + 2, a=2 / 3, b=1 / 12)
        F = (F_stencil @ self.F.flatten()).reshape(self.N + 2, self.N + 2)
        F_internal = F[1:-1, 1:-1]
        G = fps(self.h ** 2 * F_internal, eigval)
        # TODO: better to use F, instead of F_stencil @ G
        G_stencil = five_point_stencil(self.N, a=2 / 3, b=1 / 12)
        G = (G_stencil @ G.flatten()).reshape(self.N, self.N)
        U = fps(self.h ** 2 * G, eigval)
        return U


######################
### Let's do this! ###
######################
# u, f = problems.get_poisson_sin_problem()
# p = PoissonSolver(
#     f=f,
#     N=9,
#     anal_u=u,
#     store_solution=True,
# )
# p.solve()
# print(p.errors)

# u, f = problems.get_biharmonic()
# b = BiharmonicSolver(
#     f=f,
#     N=9,
#     anal_u=u,
#     store_solution=True,
# )
# b.solve()
# plt.imshow(b.U5)
# plt.show()

Ns = np.geomspace(8, 256, 8, dtype=int)
u, f = problems.get_biharmonic()
errors = {"five": [], "nine": []}
for N in Ns:
    solver = BiharmonicSolver(f, N=N, anal_u=u, store_solution=False)
    solver.solve()
    for error in errors:
        errors[error].append(solver.errors[error])
h = 1 / (Ns + 2 - 1)
plt.loglog(Ns, errors["five"], label="five")
plt.loglog(Ns, errors["nine"], label="nine")
plt.loglog(Ns, h ** 2, linestyle="dashed", label="h2")
plt.loglog(Ns, h ** 4, linestyle="dashed", label="h4")
plt.legend()
plt.show()
