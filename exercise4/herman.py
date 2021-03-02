#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def theta_method(t, A, U0, theta):
    N_time = np.shape(t)[0]
    N_space = np.shape(U0)[0]
    U = np.empty((N_time, N_space))
    U[0] = U0
    I = np.identity(N_space)

    for n in range(1, N_time):
        dt = t[n] - t[n-1]
        U[n] = np.linalg.solve(I - theta*dt*A, np.dot((I + (1-theta)*dt*A), U[n-1]))

    return U

def forward_euler(t, A, U0):
    return theta_method(t, A, U0, theta=0)

def backward_euler(t, A, U0):
    return theta_method(t, A, U0, theta=1)

def crank_nicholson(t, A, U0):
    return theta_method(t, A, U0, theta=1/2)

def animate_wave(x, U, t):
    def animate(i):
        ax.clear()
        ax.plot(x, U[i,:])

    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=20)
    plt.show()

def main():
    N = 50
    x, h = np.linspace(-1, +1, N, retstep=True)
    N = np.shape(x)[0]

    A = np.zeros((N, N))
    stencil1 = np.array([-1, 0, +1]) / h
    stencil3 = np.array([-1/8, 0, +3/8, 0, -3/8, 0, +1/8]) / h**3
    relinds1 = [i - (len(stencil1)-1)//2 for i in range(0, len(stencil1))]
    relinds3 = [i - (len(stencil3)-1)//2 for i in range(0, len(stencil3))]
    print(relinds1)
    print(relinds3)
    for i in range(0, N):
        inds1 = [(i + relind1) % N for relind1 in relinds1]
        inds3 = [(i + relind3) % N for relind3 in relinds3]
        A[i,inds1] -= (1+np.pi**2) * stencil1
        A[i,inds3] -= stencil3

    U0 = np.sin(np.pi*x)

    t = np.linspace(0, 2, 200)
    U = crank_nicholson(t, A, U0)

    animate_wave(x, U, t)

main()
