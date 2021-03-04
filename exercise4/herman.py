#!/usr/bin/python3

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

def grid_is_uniform(x, eps=1e-10):
    dx = x[1] - x[0]
    return np.max(dx - (x[1:] - x[:-1])) < eps

def theta_method(t, A, U0, theta):
    dt = t[1] - t[0]
    assert grid_is_uniform(t), "Time step not constant"

    N_time = t.shape[0]
    N_space = U0.shape[0]
    U = np.empty((N_time, N_space))
    U[0] = U0
    I = np.identity(N_space)

    # Solve matrix system M1 @ U[n] == M2 @ U[n-1] at each time step
    # The matrices M1 and M2 are constant,
    # so optimize by LU-factorizing M1 to solve the system for many different RHSes
    M1 = I - theta*dt*A
    M2 = I + (1-theta)*dt*A
    lu, piv = sp.linalg.lu_factor(M1)
    for n in range(1, N_time):
        dt = t[n] - t[n-1]
        U[n] = sp.linalg.lu_solve((lu, piv), M2 @ U[n-1])
    return U

def solve_analytical(x, t):
    t = np.array(t)
    return np.sin(np.pi*(x-t.reshape(-1,1)))

def solve_numerical(x, t, method="crank-nicholson"):
    assert grid_is_uniform(x), "Space step not uniform"
    assert grid_is_uniform(t), "Time step not uniform"

    N = x.shape[0]
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    print(f"M = {N}, dt/dx^2 = {dt/dx**2}")

    A = np.zeros((N, N))
    stencil1 = np.array([-1, 0, +1]) / (2*dx) # 1st derivative
    stencil3 = np.array([-1/8, 0, +3/8, 0, -3/8, 0, +1/8]) / dx**3 # 3rd derivative
    relinds1 = [i - (len(stencil1)-1)//2 for i in range(0, len(stencil1))]
    relinds3 = [i - (len(stencil3)-1)//2 for i in range(0, len(stencil3))]
    for i in range(0, N):
        # impose periodic BCs by wrapping stencil coefficients around the matrix
        inds1 = [(i + relind1) % N for relind1 in relinds1]
        inds3 = [(i + relind3) % N for relind3 in relinds3]
        A[i,inds1] -= (1+np.pi**2) * stencil1
        A[i,inds3] -= stencil3

    U0 = solve_analytical(x, [t[0]])[0] # sin(pi*x)

    if method == "forward-euler":
        return theta_method(t, A, U0, 0)
    elif method == "backward-euler":
        return theta_method(t, A, U0, 1)
    elif method == "crank-nicholson":
        return theta_method(t, A, U0, 1/2)
    else:
        raise(f"Unknown method \"{method}\"")

def animate_solution(x, t, u1, u2):
    fig, ax = plt.subplots()

    # set constant limits with room to show both solutions at all times
    ymin = np.min((np.min(u1), np.min(u2)))
    ymax = np.max((np.max(u1), np.max(u2)))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x[0], x[-1])

    graph1, = ax.plot([], [])
    graph2, = ax.plot([], [])
    def animate(i):
        graph1.set_data(x, u1[i,:])
        graph2.set_data(x, u2[i,:])

    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=0)
    plt.show()

def convergence_plot():
    Ms = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    errs = []
    for M in Ms:
        x, dx = np.linspace(-1, +1, M, retstep=True)
        t, dt = np.linspace(0, 1, 200, retstep=True)
        u = solve_analytical(x, t)[-1] # u(t=1)
        U = solve_numerical(x, t)[-1] # U(t=1)
        err = np.linalg.norm(u-U, 2) / np.linalg.norm(u, 2)
        errs.append(err)

    plt.loglog(Ms, errs)
    plt.show()

def main():
    N = 400
    x, dx = np.linspace(-1, +1, N, retstep=True)
    t, dt = np.linspace(0, 1, 500, retstep=True)
    print(f"dt/dx^2 = {dt/dx**2}") # stability depends on this number (?)

    u = solve_analytical(x, t)
    U = solve_numerical(x, t)
    animate_solution(x, t, U, u)

main()
# convergence_plot()
