import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation


def backward_euler(f, M, N, t_end, g_0, g_1, bc="d"):
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
    
    if bc == "d":
        x = np.linspace(h, 1-h, M)  # Not including x = 0 if there are
                                    # Dirichlet boundary condtitions
    elif bc == "n":                 # For Neumann boundary conditions,
        x = np.linspace(0, 1, M+2)  # x = 0 must be included.

    t = np.linspace(0, t_end, N)
    sol = np.zeros((N, len(x)))
    sol[0] = f(x.copy())

    main_diag = np.full(len(x), 1+2*r)
    off_diag_upper = np.full(len(x)-1, -r)
    off_diag_lower = np.full(len(x)-1, -r)

    if bc == "d":
        A = np.diag(off_diag_lower, -1) + np.diag(off_diag_upper, 1) + np.diag(main_diag, 0)
        for i in range(N-1):
            b = sol[i]
            b[0] += r*g_0(f, t[i+1])
            b[-1] += r*g_1(f, t[i+1])
            sol[i+1] = linalg.solve(A, b)
        x = np.insert(x, 0, 0)
        x = np.insert(x, len(sol[0]), 1)
        sol = np.insert(sol, 0, f(0), axis=1)
        sol = np.insert(sol, len(sol[0]), f(1), axis=1)
    elif bc == "n":
        main_diag[0] = 1-r
        main_diag[-1] = 1+r
        off_diag_upper[0] = r
        A = np.diag(off_diag_lower, -1) + np.diag(off_diag_upper, 1) + np.diag(main_diag, 0)
        for i in range(N-1):
            b = sol[i]
            b[0] -= 2 * r*r * h * g_0(f, t[i+1])
            b[-1] += 2 * r*r * h * g_1(f, t[i+1])
            sol[i+1] = linalg.solve(A, b)
    return x, sol

def crank_nicolson(f, M, N, t_end, g_0, g_1, bc="d"):
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
    
    if bc == "d":
        x = np.linspace(h, 1-h, M)    # Not including x = 0 if there are
                                    # Dirichlet boundary condtitions
    elif bc == "n":                 # For Neumann boundary conditions,
        x = np.linspace(0, 1, M+2)  # x = 0 must be included.

    t = np.linspace(0, t_end, N)
    sol = np.zeros((N, len(x)))
    sol[0] = f(x.copy())

    main_diag = np.full(len(x), 1+r)
    off_diag_upper = np.full(len(x)-1, -r/2)
    off_diag_lower = np.full(len(x)-1, -r/2)

    if bc == "d":
        A = np.diag(off_diag_lower, -1) + np.diag(off_diag_upper, 1) + np.diag(main_diag, 0)
        for i in range(N-1):
            # b is the right hand side of the equation Au=b.
            b = np.zeros(len(x))
            b[1:-1] = 0.5*r*sol[i,:-2] + (1-r)*sol[i,1:-1] + 0.5*r*sol[i,2:]
            b[0] += 0.5*r*g_0(f, t[i+1])
            b[-1] += 0.5*r*g_1(f, t[i+1])
        
            # sol[i,:] is the unknown vector u at time step i.
            sol[i+1] = linalg.solve(A, b)

        # Insert values for f(0) and f(1) in sol and x, for all times.
        x = np.insert(x, 0, 0)
        sol = np.insert(sol, 0, f(0), axis=1)
        x = np.insert(x, len(sol[i]), 1)
        sol = np.insert(sol, len(sol[i]), f(1), axis=1)
        print(sol[i])
        print(x)
    elif bc == "n":
        main_diag[0] = 1+r/2
        main_diag[-1] = 1+r/2
        off_diag_upper[0] = -r/2
        off_diag_lower[-1] = -r/2
        A = np.diag(off_diag_lower, -1) + np.diag(off_diag_upper, 1) + np.diag(main_diag, 0)
        for i in range(N-1):
            b = np.zeros(len(x))
            b[1:-1] = 0.5*r*sol[i,:-2] + (1-r)*sol[i,1:-1] + 0.5*r*sol[i,2:]
            b[0] = sol[i,0] + 0.5*r*(sol[i,1] -sol[i,0]) - 2*r*h*g_0(f, t[i+1])
            b[-1] = sol[i,-1] + 0.5*r*(sol[i,-2] - sol[i,-1]) - 2*r*h*g_1(f, t[i+1])
            sol[i+1] = linalg.solve(A, b)
    return x, sol

def animate(i, x, U, curve):
    curve.set_data(x, U[i])
    return curve 

def animate_time_development(x, U, title):
    '''
    Animates values of U developing in N time steps.
    Variables:
        x:  np.array of x values in which function is evaluated.
        U:  np.array with dimensions (N, M) holding 
            M function values in N time steps.
    '''
    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(0,f(1)))
    ax.set_title(title)
    curve, = ax.plot(x, U[0])
    anim = animation.FuncAnimation(fig, animate, interval=200, fargs=(x, U, curve))
    return anim

def f(x):
    return 2*np.pi*x - np.sin(2*np.pi*x)

def g(x):
    return 1-np.abs(2*x-1)

def burger(x):
    return np.exp(-400*(x - 0.5)**2)

def g_0(f, t):
    return f(0)

def g_1(f, t):
    return f(1)


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


#x, U_1 = solve_order_1(f, m, h, alpha, sigma)
#x, U_2 = solve_order_2(f, m, h, alpha, sigma)
#x, U_2 = solve_order_2(v, m, h, alpha, sigma)
#plt.plot(x, U_euler[0])
#plt.plot(x, U_euler[1])
#plt.plot(x, U_euler[2])
#plt.show()

m = 100
N = 100
t_end = 1
h = 1/(m+1)
alpha = 0
sigma = 0

x_be, U_be = backward_euler(f, m, N, t_end, g_0, g_1, bc="n")
x_cn, U_cn = crank_nicolson(f, m, N, t_end, g_0, g_1, bc="n")


anim_be = animate_time_development(x_be, U_be, "Soltuion with backward Euler")
plt.show()

anim_cn = animate_time_development(x_cn, U_cn, "Soltuion with Crank-Nicolson")
plt.show()
    
