import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import dst, idst
from functools import partial

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


def five_point_stencil(N, a=-2, b=1):
    """Generate a five point stencil"""
    K = TST(N, a=a, b=b)
    I = scipy.sparse.eye(N)
    return scipy.sparse.kron(I, K) + scipy.sparse.kron(K, I)


def five_point_eigenval(N, k, l):
    """Eigenvalue of the five point stencil, for eigenvector k,l"""
    return 2 * (np.cos(k * np.pi / (N + 1)) + np.cos(l * np.pi / (N + 1))) - 4


def nine_point_stencil(N):
    """Generate a nine point stencil"""
    five_point = five_point_stencil(N, a=-5/3, b=2/3)
    SIGMA = scipy.sparse.diags([np.ones(N - 1), np.ones(N - 1)], [-1, 1])
    diagonal_points = scipy.sparse.kron(SIGMA, SIGMA) / 6

    return five_point + diagonal_points


def nine_point_eigenval(N, k, l):
    return (
        -10 / 3
        + 4 / 6 * (np.cos(k * np.pi / (N + 1)) * np.cos(l * np.pi / (N + 1)))
        + 4 / 3 * (np.cos(k * np.pi / (N + 1)) + np.cos(l * np.pi / (N + 1)))
    )



def fps(F, eigval_array, **kwargs):
    """Fast poisson solver.
    eigval_array are the eigenvalues corresponding to the stencil to solve for
    kwargs are passed on to the dst transform."""
    return idst2D(dst2D(F, **kwargs) / eigval_array, **kwargs)


def get_eigval_array(N, eigval_func):
    k = l = np.arange(1, N+1)
    kk, ll = np.meshgrid(k, l)
    return eigval_func(N, kk, ll)


def get_mesh(N):
    x = y = np.linspace(0, 1, N + 2)[1:-1]
    xx, yy = np.meshgrid(x, y)
    return xx, yy



def test_order():
    def f(x, y):
        return np.sin(x * np.pi) * np.sin(y * np.pi)
    # Ns = np.geomspace(20, 1000, 40, dtype=int)
    Ns = np.geomspace(8, 256, num=6, dtype=int)
    errors_five = []
    errors_nine = []
    kwargs = {"type": 1, "norm": "ortho"}
    for N in Ns:
        h = 1 / (N - 1)
        F = f(*get_mesh(N))
        U5 = fps(
            h**2  * F,
            get_eigval_array(N, five_point_eigenval),
            **kwargs
        )

        F_stencil = five_point_stencil(N, a=1/3, b=1/12)
        RHS9 = (F_stencil @ F.flatten()).reshape(N, N)
        RHS9 = F + h**2 * (five_point_stencil(N)/12 @ F.flatten()).reshape(N, N)
        U9 = fps(
            RHS9,
            h**-2 * get_eigval_array(N, nine_point_eigenval),
            **kwargs
        )
        U9 = U9
        # U5 = scipy.sparse.linalg.spsolve(five_point_stencil(N) / h**2, F.flatten()).reshape(N, N)
        U9 = scipy.sparse.linalg.spsolve(nine_point_stencil(N)/h**2, RHS9.flatten()).reshape(N, N)

        anal = F / -(2 * np.pi ** 2)
        diff5 = (U5 - anal).flatten()
        diff9 = (U9 - anal).flatten()
        order = np.inf
        errors_five.append(
            np.linalg.norm(diff5, ord=order)
            / np.linalg.norm(anal, ord=order)
        )

        errors_nine.append(
            np.linalg.norm(diff9, ord=order)
            / np.linalg.norm(anal, ord=order)
        )
    plt.loglog(Ns, errors_five, '-x', label="five")
    #plt.loglog([1e1, 1e2], [1e-1, 1e-2])
    plt.loglog(Ns, errors_nine, '-x', label="nine")
    plt.loglog(Ns, (1/(Ns - 1))**2 * errors_nine[0] / (1/(Ns[0] - 1))**2, label="h^2")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.grid()
    plt.show()


def exercise_h():
    def f(x, y):
        return (
            (np.sin(np.pi*x) * np.sin(np.pi*y))**4
            *
            np.exp(-(x-0.5)**2 - (y-0.5)**2)
        )
    N = 100
    F = f(*get_mesh(N))
    h = 1 / (N - 1)
    G = fps(
        h**2 * F,
        get_eigval_array(N, five_point_eigenval),
    )
    U5 = fps(
        h**2 * G,
        get_eigval_array(N, five_point_eigenval),
    )

    plt.subplot(121)
    plt.imshow(U5)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(-G)
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    test_order()
    # exercise_h()
