import numpy as np
import numpy.linalg as linalg

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

