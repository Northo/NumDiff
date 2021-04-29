import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import dst, idst
from scipy.special import erf
from functools import partial
from scipy.interpolate import NearestNDInterpolator
import time
from tqdm import tqdm
import sympy

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


def five_point_solve(F, use_fps=True, **fps_kwargs):
    N = F.shape[0]
    h = 1 / (N + 2 - 1)
    if not fps_kwargs:
        fps_kwargs = fps_kwargs = {"type": 1, "norm": "ortho"}
    if use_fps:
        U5 = fps(
            h**2  * F,
            get_eigval_array(N, five_point_eigenval),
            **fps_kwargs
        )
    else:
        U5 = scipy.sparse.linalg.spsolve(
            five_point_stencil(N),
            h**2 * F.flatten(),
        ).reshape(N, N)
    return U5



def nine_point_solve(F, use_fps=True, **fps_kwargs):
    N = F.shape[0]
    h = 1 / (N + 2 - 1)
    F_stencil = five_point_stencil(N, a=2/3, b=1/12)
    F9 = (F_stencil @ F.flatten()).reshape(N, N)
    if not fps_kwargs:
        fps_kwargs = fps_kwargs = {"type": 1, "norm": "ortho"}
    if use_fps:
        U9 = fps(
            h**2  * F9,
            get_eigval_array(N, nine_point_eigenval),
            **fps_kwargs
        )
    else:
        U9 = scipy.sparse.linalg.spsolve(
            nine_point_stencil(N),
            h**2 * F9.flatten(),
        ).reshape(N, N)
    return U9



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
#     for i in [1, 2, 4]:
#         plt.loglog(Ns, hs**i, label=f"$h^{i}$")
    for error in errors:
        if error == "Ns": continue
        plt.loglog(Ns, errors[error],  '-x', label=error)
    plt.legend()
    plt.show()


def demonstrate_order(plot=False, order=np.inf, relative=False):
    def errfunc(approx, anal):
        if relative:
            return (
                np.linalg.norm(anal.flatten() - approx.flatten(), ord=order)
                / np.linalg.norm(anal.flatten(), ord=order)
            )
        else:
            return np.linalg.norm(anal.flatten() - approx.flatten(), ord=order)

    def f(x, y, k=1, l=1):
        """Manufactured solution"""
        return np.sin(x * k * np.pi) * np.sin(y * l * np.pi)
    Ns = np.geomspace(8, 256, num=6, dtype=int)
    k = 3
    l = 4
    errors = {
        "Ns": Ns,
        "five": [],
        "five_roof": [],
        "nine": [],
        "nine_roof": [],
    }
    fps_kwargs = {"type": 1, "norm": "ortho"}
    for N in Ns:
        xx, yy, h = get_mesh(N, reth=True)
        F = f(xx, yy, k, l)
        U_anal = F / (-np.pi**2 * (k**2 + l**2))
        ### Five point stencil ###
        U5 = five_point_solve(
            F,
            use_fps=True,
            **fps_kwargs
        )
        ### Nine point stecnil ###
        U9 = nine_point_solve(
            F,
            use_fps=True,
            **fps_kwargs
        )

        errors["five"].append(errfunc(U5, U_anal))
        errors["five_roof"].append(
            # 1/8 * 1/12 * h**2 * (k*np.pi)**2 * (l*np.pi)**2 / ((np.pi*k)**2 + (np.pi*l)**2)
             h**2 * 0.1
        )
        errors["nine"].append(errfunc(U9, U_anal))
        errors["nine_roof"].append(
            h**4 * 0.1
        )

    if plot:
        plot_errors(Ns, errors)

    return errors


def demonstrate_order_biharmonic(plot=False, order=np.inf, relative=False):
    def errfunc(approx, anal):
        if relative:
            return (
                np.linalg.norm(anal.flatten() - approx.flatten(), ord=order)
                / np.linalg.norm(anal.flatten(), ord=order)
            )
        else:
            return np.linalg.norm(anal.flatten() - approx.flatten(), ord=order)

    def f(x, y, k=1, l=1):
        """Manufactured solution"""
        return np.sin(x * k * np.pi) * np.sin(y * l * np.pi)
    Ns = np.geomspace(8, 256, num=6, dtype=int)
    k = 3
    l = 4
    errors = {
        "Ns": Ns,
        "five": [],
        "five_roof": [],
        "nine": [],
        "nine_roof": [],
    }
    fps_kwargs = {"type": 1, "norm": "ortho"}
    for N in Ns:
        xx, yy, h = get_mesh(N, reth=True)
        F = f(xx, yy, k, l)
        U_anal = F / ((k*np.pi)**2 + (l*np.pi)**2)**2
        ### Five point stencil ###
        G5 = five_point_solve(
            F,
            use_fps=True,
            **fps_kwargs
        )
        U5 = five_point_solve(
            G5,
            use_fps=True,
            **fps_kwargs
        )
        ### Nine point stecnil ###
        G9 = nine_point_solve(
            F,
            use_fps=True,
            **fps_kwargs
        )
        U9 = nine_point_solve(
            G9,
            use_fps=True,
            **fps_kwargs
        )

        errors["five"].append(errfunc(U5, U_anal))
        errors["five_roof"].append(
            # 1/8 * 1/12 * h**2 * (k*np.pi)**2 * (l*np.pi)**2 / ((np.pi*k)**2 + (np.pi*l)**2)
             h**2 * 0.1
        )
        errors["nine"].append(errfunc(U9, U_anal))
        errors["nine_roof"].append(
            h**4 * 0.1
        )

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


def __integral(m):
    return (
        1/16*np.sqrt(np.pi)*(
            erf(-2*1j*np.pi + 1/2*1j*np.pi*m + 1/2)*np.exp(4*np.pi**2*m)*np.sin(1/2*np.pi*m)
            - erf(-2*1j*np.pi + 1/2*1j*np.pi*m - 1/2)*np.exp(4*np.pi**2*m)*np.sin(1/2*np.pi*m)
            + 4*erf(-1j*np.pi + 1/2*1j*np.pi*m + 1/2)*np.exp(3*np.pi**2*m + 3*np.pi**2)*np.sin(1/2*np.pi*m)
            - 4*erf(-1j*np.pi + 1/2*1j*np.pi*m - 1/2)*np.exp(3*np.pi**2*m+ 3*np.pi**2)*np.sin(1/2*np.pi*m)
            + 6*erf(1/2*1j*np.pi*m + 1/2)*np.exp(2*np.pi**2*m + 4*np.pi**2)*np.sin(1/2*np.pi*m)
            - 6*erf(1/2*1j*np.pi*m - 1/2)*np.exp(2*np.pi**2*m + 4*np.pi**2)*np.sin(1/2*np.pi*m)
            + 4*erf(1j*np.pi + 1/2*1j*np.pi*m + 1/2)*np.exp(np.pi**2*m + 3*np.pi**2)*np.sin(1/2*np.pi*m)
            - 4*erf(1j*np.pi + 1/2*1j*np.pi*m - 1/2)*np.exp(np.pi**2*m + 3*np.pi**2)*np.sin(1/2*np.pi*m)
            + erf(2*1j*np.pi + 1/2*1j*np.pi*m + 1/2)*np.sin(1/2*np.pi*m)
            - erf(2*1j*np.pi + 1/2*1j*np.pi*m - 1/2)*np.sin(1/2*np.pi*m)
        )*np.exp(-1/4*np.pi**2*m**2 - 2*np.pi**2*m - 4*np.pi**2)
    )

    # return (
    #     48*(
    #         np.pi**9*m**5*np.e**2
    #         - 20*(np.pi**9*np.e**2 + 2*np.pi**7*np.e**2)*m**3
    #         - (np.pi**9*m**5 - 20*(np.pi**9 + 2*np.pi**7)*m**3 + 16*(4*np.pi**9 + 15*np.pi**7 + 5*np.pi**5)*m)*(-1)**m
    #         + 16*(4*np.pi**9*np.e**2 + 15*np.pi**7*np.e**2 + 5*np.pi**5*np.e**2)*m)
    #     /
    #     (
    #         np.pi**10*m**10*np.e**3
    #         - 20*(2*np.pi**10*np.e**3 - np.pi**8*np.e**3)*m**8
    #         + 16384*np.pi**8*np.e**3
    #         + 16*(33*np.pi**10*np.e**3 - 20*np.pi**8*np.e**3 + 10*np.pi**6*np.e**3)*m**6
    #         + 40960*np.pi**6*np.e**3
    #         - 320*(8*np.pi**10*np.e**3 - 7*np.pi**8*np.e**3 - 2*np.pi**4*np.e**3)*m**4
    #         + 33792*np.pi**4*np.e**3
    #         + 256*(16*np.pi**10*np.e**3 + 35*np.pi**6*np.e**3 + 20*np.pi**4*np.e**3 + 5*np.pi**2*np.e**3)*m**2
    #         + 10240*np.pi**2*np.e**3 + 1024*np.e**3
    #     )
    # )


def analytical_solution_fourier(m, n):
    return __integral(m) * __integral(n)


def get_analytical_solution(terms=5):
    m = n = np.arange(1, terms+1)
    mm, nn = np.meshgrid(m, n)
    fourier_coeff = analytical_solution_fourier(mm, nn) / ((mm * np.pi)**2 + (nn * np.pi)**2)**2
    def analytical_solution(x, y):
        sines = (
            np.sin(mm * np.pi * x)
            * np.sin(nn * np.pi * y)
        )
        return np.sum(sines * np.real(fourier_coeff))
    return np.vectorize(analytical_solution)


@np.vectorize
def new_analytical_solution(x, y):
    return (
        4*(
            6*np.pi**4*np.exp(x + y)*np.sin(np.pi*x)**4
            - 8*(
                4*np.pi*y**3*np.exp(x)*np.sin(np.pi*x)**4 - 6*np.pi*y**2*np.exp(x)*np.sin(np.pi*x)**4 - 6*np.pi**3*np.exp(x)*np.sin(np.pi*x)**2 + 4*(2*np.pi**2*x - np.pi**2)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 + (3*np.pi + 16*np.pi**3 - 2*np.pi*x**2 + 2*np.pi*x)*np.exp(x)*np.sin(np.pi*x)**4 + 4*(3*np.pi**3*np.exp(x)*np.sin(np.pi*x)**2 - 2*(2*np.pi**2*x - np.pi**2)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (np.pi + 8*np.pi**3 - np.pi*x**2 + np.pi*x)*np.exp(x)*np.sin(np.pi*x)**4)*y
            )*np.cos(np.pi*y)*np.exp(y)*np.sin(np.pi*y)**3
            + (
                4*y**4*np.exp(x)*np.sin(np.pi*x)**4 - 8*y**3*np.exp(x)*np.sin(np.pi*x)**4 - 8*(3*np.pi + 4*np.pi*x**3 + 16*np.pi**3 - 6*np.pi*x**2 - 4*(np.pi + 8*np.pi**3)*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 + (256*np.pi**4 + 4*x**4 - 8*(16*np.pi**2 + 1)*x**2 - 8*x**3 + 64*np.pi**2 + 4*(32*np.pi**2 + 3)*x + 1)*np.exp(x)*np.sin(np.pi*x)**4 + 6*np.pi**4*np.exp(x) - 24*(2*np.pi**3*x - np.pi**3)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x) - 12*(13*np.pi**4 - 6*np.pi**2*x**2 + 6*np.pi**2*x + 2*np.pi**2)*np.exp(x)*np.sin(np.pi*x)**2 + 8*(2*(np.pi - 2*np.pi*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (16*np.pi**2 - x**2 + x + 1)*np.exp(x)*np.sin(np.pi*x)**4 + 3*np.pi**2*np.exp(x)*np.sin(np.pi*x)**2)*y**2 - 4*(4*(np.pi - 2*np.pi*x)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3 - (32*np.pi**2 - 2*x**2 + 2*x + 3)*np.exp(x)*np.sin(np.pi*x)**4 + 6*np.pi**2*np.exp(x)*np.sin(np.pi*x)**2)*y
            )*np.exp(y)*np.sin(np.pi*y)**4
            - 24*(2*np.pi**3*y*np.exp(x)*np.sin(np.pi*x)**4 - np.pi**3*np.exp(x)*np.sin(np.pi*x)**4)*np.cos(np.pi*y)*np.exp(y)*np.sin(np.pi*y)
            + 12*(
                6*np.pi**2*y**2*np.exp(x)*np.sin(np.pi*x)**4
                - 6*np.pi**2*y*np.exp(x)*np.sin(np.pi*x)**4
                + 6*np.pi**4*np.exp(x)*np.sin(np.pi*x)**2
                - 4*(2*np.pi**3*x - np.pi**3)*np.cos(np.pi*x)*np.exp(x)*np.sin(np.pi*x)**3
                - (13*np.pi**4 - 2*np.pi**2*x**2 + 2*np.pi**2*x + 2*np.pi**2)*np.exp(x)*np.sin(np.pi*x)**4
            )*np.exp(y)*np.sin(np.pi*y)**2
        )*np.exp(-x**2 - y**2 - 1/2)
    )


def exercise_h(
        ord=np.inf,
        use_fps=True,
        relative=False,
        nminmax=(8, 256),
        numN=8,
        stencil="nine",
):

    ### The updated formulation ###
    f = new_analytical_solution
    def anal(x,y):
        return (
            (np.sin(np.pi*x) * np.sin(np.pi*y))**4
            *
            np.exp(-(x-0.5)**2 - (y-0.5)**2)
        )

    ### Any formulation, solved with sympy ###
    # x, y = sympy.var("x, y")
    # u = (
    #     (sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y))**4
    #     * sympy.exp(-(x-sympy.Rational(1,2))**2 - (y-sympy.Rational(1,2))**2)
    # )
    # u = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y) * sympy.exp(-(x-sympy.Rational(1,2))**2 - (y-sympy.Rational(1,2))**2)
    # #    u = (sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)) * sympy.exp(x**2)
    # f = u.diff(x, 4) + u.diff(y, 4) + 2 * u.diff(x, 2).diff(y, 2)
    # f = f.simplify()
    # print("f: ", f)
    # f = sympy.lambdify([x,y], f, "numpy")
    # anal = sympy.lambdify([x,y], u, "numpy")


    ### The first formulation, which was wrong ###
    # def f(x, y):
    #     return (
    #         (np.sin(np.pi*x) * np.sin(np.pi*y))**4
    #         *
    #         np.exp(-(x-0.5)**2 - (y-0.5)**2)
    #     )
    # anal = get_analytical_solution(terms=8)

    ### Simple test case ###
    # def f(x, y):
    #     return (
    #         np.sin(np.pi * x) * np.sin(np.pi * y * 3)
    #         + np.sin(np.pi * x * 4) * np.sin(np.pi * y * 3)
    #     )

    # def anal(x, y):
    #     return (
    #         np.sin(np.pi * x) * np.sin(np.pi * y * 3) / ((np.pi)**2 + (np.pi * 3)**2)**2
    #         + np.sin(np.pi * x * 4) * np.sin(np.pi * y * 3) / ((np.pi*4)**2 + (np.pi * 3)**2)**2
    #     )

    def errfunc(U, u, ord=np.inf, relative=False):
        U = U.flatten()
        u = u.flatten()
        if relative:
            return (
                np.linalg.norm(
                    U - u,
                    ord=ord,
                )
                / np.linalg.norm(
                    u,
                    ord=ord,
                )
            )
        else:
            return np.linalg.norm(
                U - u,
                ord=ord,
            )

    # N = 1000
    # xx, yy, h = get_mesh(N, reth=True)
    # F = f(xx, yy)
    # G_anal = nine_point_solve(F)
    # U9_anal = nine_point_solve(G_anal)
    # anal = NearestNDInterpolator(list(zip(xx.flatten(), yy.flatten())), U9_anal.flatten())
    errors = []
    comp_time = []
    Ns = np.geomspace(*nminmax, numN, dtype=int)
    use_fps = use_fps

    if stencil == "nine":
        solver = nine_point_solve
    elif stencil == "five":
        solver = five_point_solve
    else:
        raise("Invalid stencil chosen")
    print(solver)
    for N in Ns:
        print(f"Running N = {N}...", end="")
        xx, yy, h = get_mesh(N, reth=True)
        F = f(xx, yy)
        print(" F ... ", end="")
        start_time = time.time()
        F_stencil = five_point_stencil(N, a=2/3, b=1/12)
        F9 = (F_stencil @ F.flatten()).reshape(N, N)
        fps_kwargs = fps_kwargs = {"type": 1, "norm": "ortho"}
        G9 = fps(
            h**2  * F9,
            get_eigval_array(N, nine_point_eigenval),
            **fps_kwargs
        )
        U9 = fps(
            h**2  * (G9 + h**2/12 * F),
            get_eigval_array(N, nine_point_eigenval),
            **fps_kwargs
        )


        # G = solver(F, use_fps=use_fps)
        # U9 = solver(G, use_fps=use_fps)
        comp_time.append(time.time() - start_time)
        errors.append(
            errfunc(U9, anal(xx, yy), ord=ord, relative=relative)
        )
        print("O")

    # Save solution
    # data = np.array([d.flatten() for d in [xx, yy, U9]]).T
    # header = "x y U"
#    np.savetxt(f"biharmonic_solution_{Ns[-1]}.dat", data, header=header, comments="")
    plt.subplot(121)
    plt.imshow(anal(xx, yy))#, vmin=-1e-3, vmax=1e-3)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(U9)#, vmin=-1e-3, vmax=1e-3)
    plt.colorbar()
    plt.show()

    h0 = h
    h = 1 / (Ns + 2 - 1)
    print(h, h[0], h[1], h[-1])
    plt.loglog(Ns, errors, '-x', label=stencil)
    plt.loglog(Ns, h**2, label="h^2")
    plt.loglog(Ns, h**4, label="h^4")
    plt.legend()
    plt.show()

    plt.loglog(Ns, comp_time, '-x')
    plt.show()

    return errors, comp_time, Ns


def plot_fourier(m_max):
    m = n = np.arange(1, m_max+1)
    mm, nn = np.meshgrid(m, n)
    coef = (
        __integral(mm) * __integral(nn)/
        ((mm * np.pi)**2 + (nn * np.pi)**2 )**2
    )
    plt.imshow(
        np.real(coef)
    )
    print(coef[0])
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    # errors = demonstrate_order_biharmonic(True, order=2, relative=True)
    # data = [errors[error] for error in errors]
    # header = ' '.join(errors.keys())
    # np.savetxt(
    #     "order_rel_norm2.dat",
    #     np.vstack(data).T,
    #     header=header,
    #     comments='',
    # )
    # test_order()


    errors, comp_time, Ns = exercise_h(
        ord=2,
        use_fps=True,
        relative=True,
        nminmax=(8, 256),
        numN=8,
        stencil="nine",
    )
    data = [Ns, errors]
    header = "N error"
    np.savetxt(
        "error_bvp_rel_2_five_stencil.dat",
        np.vstack(data).T,
        header=header,
        comments='',
    )
    data = [Ns, comp_time]
    header = "N comp_time"
    np.savetxt(
        "comp_bvp.dat",
        np.vstack(data).T,
        header=header,
        comments='',
    )


    # # plot_fourier(4)
