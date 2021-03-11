#!/usr/bin/python3

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

def write_table_to_file(path, table, headers):
    np.savetxt(path, table, header=" ".join(headers), comments="")
    print(f"Wrote to {path}")

def grid_is_uniform(x, eps=1e-10):
    dx = x[1] - x[0]
    return np.max(dx - (x[1:] - x[:-1])) < eps

def theta_method(t, A, U0, theta):
    dt = t[1] - t[0]
    assert grid_is_uniform(t), "Time step not constant"

    N = t.shape[0]
    M = U0.shape[0]
    U = np.empty((N, M))
    U[0] = U0
    I = np.identity(M)

    # Solve matrix system M1 @ U[n] == M2 @ U[n-1] at each time step
    # The matrices M1 and M2 are constant,
    # so optimize by LU-factorizing M1 to solve the system for many different RHSes
    M1 = I - theta*dt*A
    M2 = I + (1-theta)*dt*A
    lu, piv = sp.linalg.lu_factor(M1)
    for n in range(1, N):
        dt = t[n] - t[n-1]
        U[n] = sp.linalg.lu_solve((lu, piv), M2 @ U[n-1])
    return U

def solve_analytical(x, t):
    t = np.array(t)
    return np.sin(np.pi*(x-t.reshape(-1,1)))

def solve_numerical(x, t, method="crank-nicholson"):
    assert grid_is_uniform(x), "Space step not uniform"
    assert grid_is_uniform(t), "Time step not uniform"

    M = x.shape[0]
    N = t.shape[0]
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    print(f"M = {M}, N = {N}, dt/dx^2 = {dt/dx**2}")

    A = np.zeros((M, M))
    stencil1 = np.array([-1, 0, +1]) / (2*dx) # 1st derivative
    stencil3 = np.array([-1/8, 0, +3/8, 0, -3/8, 0, +1/8]) / dx**3 # 3rd derivative
    relinds1 = [i - (len(stencil1)-1)//2 for i in range(0, len(stencil1))]
    relinds3 = [i - (len(stencil3)-1)//2 for i in range(0, len(stencil3))]
    for i in range(0, M):
        # impose periodic BCs by wrapping stencil coefficients around the matrix
        inds1 = [(i + relind1) % M for relind1 in relinds1]
        inds3 = [(i + relind3) % M for relind3 in relinds3]
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
        graph1.set_label(f"u1(t={t[i]:.2f})")
        graph2.set_label(f"u2(t={t[i]:.2f})")
        ax.legend(loc="upper right")

    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=0, frames=len(t))
    plt.show()

def norm_evolution():
    series = [
        {"method": "crank-nicholson", "M": 200, "N": 200},
        {"method": "forward-euler", "M": 100, "N": 100}
    ]

    for s in series:
        method = s["method"]

        x = np.linspace(-1, +1, s["M"])
        t = np.linspace(0, 1, s["N"])
        U = solve_numerical(x, t, method=method)

        N = t.shape[0]
        L2 = np.empty(N)
        for i in range(0, N):
            L2[i] = np.linalg.norm(U, 2)

        label = f"{method}"
        plt.plot(t, (L2 - np.mean(L2)) / np.mean(L2), label=label)

    plt.ylim(-0.01, +0.01) # relative error always [-1, +1]
    plt.legend()
    plt.show()

def main(animate=True, write=False, time_samples=5):
    N = 400
    x = np.linspace(-1, +1, N)
    t = np.linspace(0, 1, 500)

    u = solve_analytical(x, t)
    U = solve_numerical(x, t)

    if animate:
        animate_solution(x, t, U, u)

    if write:
        inds = np.round(np.linspace(0, len(t)-1, time_samples)).astype(int)

        cols = [x] + [u[i,:] for i in inds] + [U[i,:] for i in inds]
        headers = ["x "] + [f"{t[i]}" for i in inds]
        table = np.transpose(np.array(cols))
        write_table_to_file("../report/exercise4/timeevol.dat", table, headers)

def write_results(x, t, U, path):
    x, y = np.meshgrid(x, t)
    x, y = x.reshape(-1), y.reshape(-1)
    z = U.reshape(-1)
    print(x)
    print(y)
    print(z)
    write_table_to_file(path, np.transpose([x,y,z]), ["x t U"])

def convergence_plots():
    runs = [
        {"method": "crank-nicholson", "M": [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10], "N": [10, 20, 30, 40, 50]},
        {"method": "forward-euler", "M": [5,10,15,20,25,30], "N": [10000, 20000, 30000]},
        # {"method": "forward-euler", "M": [3, 6, 12, 24, 48], "N": [500000, 750000, 1000000]},
    ]

    for run in runs:
        run["err"] = []
        for N in run["N"]:
            run["err"].append([])
            for M in run["M"]:
                x = np.linspace(-1, +1, M)
                t = np.linspace(0, 1, N)
                u = solve_analytical(x, t)
                U = solve_numerical(x, t, method=run["method"])

                err = np.linalg.norm(u-U, 2) / np.linalg.norm(u, 2)
                run["err"][-1].append(err)

    for run in runs:
        plt.title(run["method"])
        for n, N in enumerate(run["N"]):
            plt.loglog(run["M"], run["err"][n], label=f"N={N}")
        plt.show()

    for run in runs:
        method = run["method"]
        columns = [run["M"]] + [run["err"][i] for i in range(0, len(run["N"]))]
        headers = ["M"     ] + [f"E{N}"       for N in run["N"]]

        path = f"../report/exercise4/convergence-{method}.dat"
        write_table_to_file(path, np.transpose(columns), headers)

def snapshots():
    runs = [
        {"method": "crank-nicholson", "M": [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], "N": [10, 100]},
        # {"method": "forward-euler", "M": [3, 6, 12, 24, 48], "N": [500000, 750000, 1000000]},
    ]

    for run in runs:
        method = run["method"]
        run["x"] = []
        run["U"] = []
        for N in run["N"]:
            run["x"].append([])
            run["U"].append([])
            for M in run["M"]:
                x = np.linspace(-1, +1, M)
                t = np.linspace(0, 1, N)
                u = solve_analytical(x, t)
                U = solve_numerical(x, t, method=run["method"])

                path = f"../report/exercise4/snapshot-{method}-M{M}-N{N}.dat"
                columns = [x, U[-1,:]]
                headers = ["x", "U"]
                write_table_to_file(path, np.transpose(columns), headers)

                # plt.plot(x, U[-1], label=f"N={N}")
            # plt.show()

# main(animate=False, write=True, time_samples=5)
# convergence_plots()
snapshots()
# norm_evolution()
