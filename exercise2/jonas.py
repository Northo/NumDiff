import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation


def backwards_euler(f, M, N, t_end, g_0, g_1):
    ''' 
    Solves a differential equation of the type
        d_t u = d_x**2 u, where 
        u(x, 0) = f(x),
        u(0, t) = g_0(t), and 
        u(1, t) = g_1(t)
    
    Arguments:
        f = f(x):     Initial condition.
        M:            Number of points in which solution is calculated.
        N:            Number of time steps for which solution is calculated.
        t_end:        Time at which calculation ends.
        g_0 = g_0(t): Fuction giving value of u at the boundary x = 0.
        g_1 = g_1(t): Fuction giving value of u at the boundary x = 1.
    Returns:
        x:   np.array with spacial points at which function is evaluated.
        sol: np.array of dimensions (N, M) with approximated values
             of u(x, t) in M points at N times.
    '''
    h = 1/(M+1)
    k = t_end/N
    r = k/h**2

    main_diag = np.full(M+1, 1 + 2*r)
    off_diag = np.full(M, -r)
    A = np.diag(off_diag, -1) + np.diag(off_diag, 1) + np.diag(main_diag, 0)

    x = np.linspace(h, 1, M+1)
    t = np.linspace(0, t_end, N)
    sol = np.zeros((N, M+1))
    sol[0] = f(x.copy())
    for i in range(N-1):
        b = sol[i]
        b[0] += r*g_0(t[i+1])
        b[-1] += r*g_1(t[i+1])
        sol[i+1] = linalg.solve(A, b)
    return x, sol

def f(x):
    return 2*np.pi*x - np.sin(2*np.pi*x)

def g(x):
    return 1-np.abs(2*x-1)

def g_0(t):
    return 0

def g_1(t):
    return 0

m = 100
N = 500
t_end = 1
h = 1/(m+1)
alpha = 0
sigma = 0

x, U_euler = backwards_euler(g, m, N, t_end, g_0, g_1)
#x, U_1 = solve_order_1(f, m, h, alpha, sigma)
#x, U_2 = solve_order_2(f, m, h, alpha, sigma)
#x, U_2 = solve_order_2(v, m, h, alpha, sigma)

#plt.plot(x, U_euler[0])
#plt.plot(x, U_euler[1])
#plt.plot(x, U_euler[2])
#plt.show()

def animate(i, x, U, curve):
    curve.set_data(x, U[i])
    return curve 

def animate_time_development(x, U):
    '''
    Animates values of U developing in N time steps.
    Variables:
        x:  np.array of x values in which function is evaluated.
        U:  np.array with dimensions (N, M) holding 
            M function values in N time steps.
    '''
    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(0,np.max(U)*1.1))
    curve, = ax.plot(x, U[0])
    anim = animation.FuncAnimation(fig, animate, fargs=(x, U, curve))
    plt.show()

animate_time_development(x, U_euler)
    

# The following functions solve boundary value 
# problems without time dependece, i.e. task 1.
def make_A(n, h, order):
    a = np.ones(n-1)
    b = np.full(n, -2)
    A = np.diag(a, -1) + np.diag(a, 1) + np.diag(b, 0)
    A = set_h(A, h, order)
    #print("Order =", order, "and h =", h, ".")
    #print("A = \n", A)
    return A

def set_h(A, h, order):
    m = len(A) - 1
    if order == 1:
        A[m, m] = -h
        A[m, m-1] = h
    elif order == 2:
        A[m, m] = -1.5*h
        A[m, m-1] = 2*h
        A[m, m-2] = -0.5*h
    return A

def make_b(f, x, h, alpha, sigma):
    b = f(x.copy())
    b[0] -= alpha/h**2
    b[-1] = sigma
    #print("b = \n", b)
    return b*h**2

def solve_order_1(f, m, h, alpha, sigma):
    A = make_A(m+1, h, 1) 
    x = np.linspace(h, 1, m+1)
    b = make_b(f, x, h, alpha, sigma)
    U = linalg.solve(A, b)
    return x, U

def solve_order_2(f, m, h, alpha, sigma):
    A = make_A(m+1, h, 2)
    x = np.linspace(h, 1, m+1)
    b = make_b(f, x, h, alpha, sigma)
    U = linalg.solve(A, b)
    return x, U

