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


def five_point_stencil(N, a=-4, b=1):
    """Generate a five point stencil"""
    K = TST(N, a=a/2, b=b)
    I = scipy.sparse.eye(N)
    return scipy.sparse.kron(I, K) + scipy.sparse.kron(K, I)


def five_point_eigenval(N, k, l):
    """Eigenvalue of the five point stencil, for eigenvector k,l"""
    return 2 * (np.cos(k * np.pi / (N + 1)) + np.cos(l * np.pi / (N + 1))) - 4


def nine_point_stencil(N):
    """Generate a nine point stencil"""
    five_point = five_point_stencil(N, a=-10/3, b=2/3)
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


def get_mesh(N, reth=False):
    x = y = np.linspace(0, 1, N + 2)[1:-1]
    xx, yy = np.meshgrid(x, y)
    if reth:
        h = 1 / (N+2 -1)  # N + 2 points in total, N internal
        return xx, yy, h
    else:
        return xx, yy


def plot_errors(Ns, errors):
    hs = 1 / (Ns + 2 - 1)  # N are internal points, N + 2 in total
    for i in [1, 2, 4]:
        plt.loglog(Ns, hs**i, label=f"$h^{i}$")
    for error in errors:
        plt.loglog(Ns, errors[error],  '-x', label=error)
    plt.legend()
    plt.show()


def demonstrate_order(plot=False):
    def errfunc(approx, anal):
        return np.linalg.norm(anal.flatten() - approx.flatten(), ord=np.inf)

    def f(x, y, k=1, l=1):
        """Manufactured solution"""
        return np.sin(x * k * np.pi) * np.sin(y * l * np.pi)
    Ns = np.geomspace(8, 256, num=6, dtype=int)
    k = 3
    l = 4
    errors = {
        "five": [],
        "nine": [],
    }
    fps_kwargs = {"type": 1, "norm": "ortho"}
    for N in Ns:
        xx, yy, h = get_mesh(N, reth=True)
        F = f(xx, yy, k, l)
        U_anal = F / (-np.pi**2 * (k**2 + l**2))
        ### Five point stencil ###
        U5 = fps(
            h**2  * F,
            get_eigval_array(N, five_point_eigenval),
            **fps_kwargs
        )
        ### Nine point stecnil ###
        F_stencil = five_point_stencil(N, a=2/3, b=1/12)
        F9 = (F_stencil @ F.flatten()).reshape(N, N)
        U9 = fps(
            h**2  * F9,
            get_eigval_array(N, nine_point_eigenval),
            **fps_kwargs
        )

        errors["five"].append(errfunc(U5, U_anal))
        errors["nine"].append(errfunc(U9, U_anal))

    if plot:
        plot_errors(Ns, errors)

    return errors


def test_order():
    def f(x, y, k=1, l=1):
        return np.sin(x * k * np.pi) * np.sin(y * l * np.pi)
    # Ns = np.geomspace(20, 1000, 40, dtype=int)
    Ns = np.geomspace(8, 256, num=6, dtype=int)
    errors_five = []
    errors_nine = []
    five_nine_diff = []
    k = 3
    l = 7
    kwargs = {"type": 1, "norm": "ortho"}
    for N in Ns:
        h = 1 / (N + 2 - 1)
        F = f(*get_mesh(N), k, l)
        U5 = fps(
            h**2  * F,
            get_eigval_array(N, five_point_eigenval),
            **kwargs
        )

        F_stencil = five_point_stencil(N, a=1/3, b=1/12)
        RHS9 = (F_stencil @ F.flatten()).reshape(N, N)
        # RHS9 = F + h**2 * (five_point_stencil(N)/12 @ F.flatten()).reshape(N, N)
        # RHS9 = F + (five_point_stencil(N)/12 @ F.flatten()).reshape(N, N)
        # RHS9 = F + 1/12 * h**2 * (-np.pi**2 * 2) * F
        U9 = fps(
            h**2 * RHS9,
            get_eigval_array(N, nine_point_eigenval),
            **kwargs
        )
        # U5 = scipy.sparse.linalg.spsolve(five_point_stencil(N) / h**2, F.flatten()).reshape(N, N)
        # U9 = scipy.sparse.linalg.spsolve(nine_point_stencil(N), h**2 * RHS9.flatten()).reshape(N, N)

        five_nine_diff.append(U5 - U9)
        anal = F / -((k*np.pi)**2 + (l*np.pi)**2)
        diff5 = (U5 - anal).flatten()
        diff9 = (U9 - anal).flatten()

        # See diff laplace
        # diff5 = five_point_stencil(N)/h**2 @ anal.flatten() - F.flatten()
        #diff9 = nine_point_stencil(N)/h**2 @ anal.flatten() - F.flatten() - 1/12 * five_point_stencil(N) @ F.flatten()# five_point_stencil(N, 1/3, 1/12) @ F.flatten()
        #anal = F.flatten()

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
    plt.loglog(Ns, (1/(Ns - 1))**1 * errors_nine[0] / (1/(Ns[0] - 1))**2, label="h^1")
    plt.loglog(Ns, (1/(Ns - 1))**2 * errors_nine[0] / (1/(Ns[0] - 1))**2, label="h^2")
    plt.loglog(Ns, (1/(Ns - 1))**4 * errors_nine[0] / (1/(Ns[0] - 1))**2, label="h^4")
    # plt.gca().set_aspect("equal")
    plt.legend()
    plt.grid()
    plt.show()

    x = np.linspace(0, 1, 2+Ns[-1])[1:-1]
    y = np.linspace(0, 1, 2+Ns[-1])[1:-1]
    plt.plot(x, U5[0, :], label="u5")
    plt.plot(x, U9[0, :], label="u9")
    plt.plot(x, f(x, y[0], k, l) / (-np.pi**2 * (k**2 + l**2 )), label="f")
    plt.legend()
    plt.show()

    plt.subplot(121)
    plt.imshow(U9)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(F / (-np.pi**2 * (k**2 + l**2)))
    plt.colorbar()
    plt.show()

    plt.loglog(Ns, [np.linalg.norm(x, ord=np.inf) for x in five_nine_diff])
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
    h = 1 / (N + 2- 1)
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
    demonstrate_order(plot=True)
    # test_order()
    # exercise_h()
