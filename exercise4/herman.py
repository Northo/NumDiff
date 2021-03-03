#!/usr/bin/python3

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

def theta_method(t, A, U0, theta):
    N_time = t.shape[0]
    N_space = U0.shape[0]
    U = np.empty((N_time, N_space))
    U[0] = U0
    I = np.identity(N_space)

    M1, M2, lu, piv = None, None, None, None
    for n in range(1, N_time):
        dt = t[n] - t[n-1]
        if n == 1 or dt - (t[n-1] - t[n-2]) > 1e-10:
            # LU-factorize matrices with equal time step
            # to solve matrix equation more efficiently for multiple inputs
            M1 = I - theta*dt*A
            M2 = I + (1-theta)*dt*A
            lu, piv = sp.linalg.lu_factor(M1)
        U[n] = sp.linalg.lu_solve((lu, piv), M2 @ U[n-1])
    return U

def forward_euler(t, A, U0):
    return theta_method(t, A, U0, theta=0)

def backward_euler(t, A, U0):
    return theta_method(t, A, U0, theta=1)

def crank_nicholson(t, A, U0):
    return theta_method(t, A, U0, theta=1/2)

def animate_solution(x, t, U, u):
    fig, ax = plt.subplots()

    ymin = np.min((np.min(U), np.min(u)))
    ymax = np.max((np.max(U), np.max(u)))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x[0], x[-1])

    graph1, = ax.plot([], [], linewidth=1, color="red")
    graph2, = ax.plot([], [], linewidth=5, color="black")

    def animate(i):
        graph1.set_data(x, U[i,:])
        graph2.set_data(x, u[i,:])

    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=0)
    plt.show()

def main():
    N = 400
    x, h = np.linspace(-1, +1, N, retstep=True)
    t, k = np.linspace(0, 1, 500, retstep=True)
    N = np.shape(x)[0]

    print(f"k/h^2 = {k/h**2}")

    A = np.zeros((N, N))
    stencil1 = np.array([-1, 0, +1]) / (2*h)
    stencil3 = np.array([-1/8, 0, +3/8, 0, -3/8, 0, +1/8]) / h**3
    relinds1 = [i - (len(stencil1)-1)//2 for i in range(0, len(stencil1))]
    relinds3 = [i - (len(stencil3)-1)//2 for i in range(0, len(stencil3))]
    for i in range(0, N):
        inds1 = [(i + relind1) % N for relind1 in relinds1]
        inds3 = [(i + relind3) % N for relind3 in relinds3]
        A[i,inds1] -= (1+np.pi**2) * stencil1
        A[i,inds3] -= stencil3

    U0 = np.sin(np.pi*x)
    U = crank_nicholson(t, A, U0)
    u = np.sin(np.pi*(x-t.reshape(-1,1)))

    animate_solution(x, t, U, u)

main()
