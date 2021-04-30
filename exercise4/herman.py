#!/usr/bin/python3

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg
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
    M1 = sp.sparse.csc_matrix(I - theta*dt*A)
    M2 = sp.sparse.csc_matrix(I + (1-theta)*dt*A)
    lu = scipy.sparse.linalg.splu(M1)
    for n in range(1, N):
        dt = t[n] - t[n-1]
        U[n] = lu.solve(M2 @ U[n-1])
    
    return U

def solve_analytical(x, t):
    t = np.array(t)
    return np.sin(np.pi*(x-t.reshape(-1,1)))

def solve_numerical(x, t, U0=None, method="crank-nicholson"):
    assert grid_is_uniform(x), "Space step not uniform"
    assert grid_is_uniform(t), "Time step not uniform"

    x = x[:-1] # remove last point (it would've been a duplicate of the first!)
    if U0 is not None:
        U0 = U0[:-1]

    M = x.shape[0]
    N = t.shape[0]
    dx = x[1] - x[0]

    print(f"M = {M}, N = {N}")

    A = np.zeros((M, M))
    for i in range(0, M):
        # impose periodic BCs by wrapping stencil coefficients around the matrix
        A[i,(i-3)%M] =                             - 1/(8*dx**3) * (-1)
        A[i,(i-1)%M] = -(1+np.pi**2)/(2*dx) * (-1) - 1/(8*dx**3) * (+3)
        A[i,(i+1)%M] = -(1+np.pi**2)/(2*dx) * (+1) - 1/(8*dx**3) * (-3)
        A[i,(i+3)%M] =                             - 1/(8*dx**3) * (+1)

    if U0 is None:
        U0 = np.sin(np.pi*x)

    U = None
    if method == "forward-euler":
        U = theta_method(t, A, U0, 0)
    elif method == "backward-euler":
        U = theta_method(t, A, U0, 1)
    elif method == "crank-nicholson":
        U = theta_method(t, A, U0, 1/2)
    else:
        raise(f"Unknown method \"{method}\"")

    U = np.append(U, np.array([U[:,0]]).T, axis=1) # add column at right boundary with same value as on the left
    return U

def animate_solution(x, t, u1, u2=None):
    fig, ax = plt.subplots()

    if u2 is None:
        ymin = np.min(u1)
        ymax = np.max(u1)
    else:
        # set constant limits with room to show both solutions at all times
        ymin = np.min((np.min(u1), np.min(u2)))
        ymax = np.max((np.max(u1), np.max(u2)))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x[0], x[-1])

    graph1, = ax.plot([], [])
    graph2, = ax.plot([], [])
    def animate(i):
        graph1.set_data(x, u1[i,:])
        graph1.set_label(f"u1(t={t[i]:.2f})")

        if u2 is not None:
            graph2.set_data(x, u2[i,:])
            graph2.set_label(f"u2(t={t[i]:.2f})")
        ax.legend(loc="upper right")

    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(t))
    plt.show()

def norm_evolution(plot=False, write=False):
    runs = [
        {"method": "forward-euler", "M": 15, "N": [30000, 40000, 50000]},
        {"method": "forward-euler", "M": 20, "N": [30000, 40000, 50000]},
        {"method": "forward-euler", "M": 25, "N": [30000, 40000, 50000]},
        {"method": "crank-nicholson", "M": 20, "N": [100, 200, 300]},
        {"method": "crank-nicholson", "M": 30, "N": [100, 200, 300]},
        {"method": "crank-nicholson", "M": 100, "N": [100, 200, 300]},
        {"method": "crank-nicholson", "M": 200, "N": [100, 200, 300]},
        {"method": "crank-nicholson", "M": 400, "N": [100, 200, 300]},
        {"method": "crank-nicholson", "M": 800, "N": [100, 200, 300]},
    ]

    for run in runs:
        method = run["method"]
        run["L2"] = []
        for N in run["N"]:
            x = np.linspace(-1, +1, run["M"])
            t = np.linspace(0, 1, N)
            # U0 = np.sin(np.pi*x) + np.sin(3*np.pi*x)
            # U0 = np.heaviside(x+0.5, 0.5) * (1 - np.heaviside(x-0.5, 0.5))
            U0 = np.exp(-10*x**2)
            U = solve_numerical(x, t, U0=U0, method=method)
            # plt.plot(x, U0)
            # plt.show()
            # animate_solution(x, t, U)

            L2 = np.linalg.norm(U, 2, axis=1) * np.sqrt(2/run["M"])
            run["L2"].append(L2)

    if plot:
        for run in runs:
            for n, N in enumerate(run["N"]):
                plt.plot(np.linspace(0, 1, N), run["L2"][n], label=run["method"])
                # plt.ylim(L2[0]-0.5, L2[0]+0.5)
        L2min = np.min([run["L2"][n][0] for n, _ in enumerate(run["N"]) for run in runs])
        L2max = np.max([run["L2"][n][0] for n, _ in enumerate(run["N"]) for run in runs])
        plt.ylim(L2min-0.5, L2max+0.5)
        plt.legend()
        plt.show()

    if write:
        # Write to file
        # headers = ["t", f"{N}"]
        # columns = [np.linspace(0, 1, 100), run["L2"][n][inds]]
        headers = []
        columns = []
        for run in runs:
            M = run["M"]
            for n, N in enumerate(run["N"]):
                inds = np.round(np.linspace(0, N-1, 100)).astype(int) # sample at 100 times
                label = run["method"] + f"-M{M}-N{N}"
                headers.append(label)
                columns.append(run["L2"][n][inds])
        path = f"../report/exercise4/norm-evolution.dat"
        write_table_to_file(path, np.transpose(columns), headers)

    # plt.ylim(-0.01, +0.01) # relative error always [-1, +1]

def timeevol(animate=True, path=None, time_samples=5, U0=None, N=100, M=800, method="crank-nicholson"):
    x = np.linspace(-1, +1, M)
    t = np.linspace(0, 1, N)

    if U0 is None:
        u = solve_analytical(x, t)
        U = solve_numerical(x, t, method=method)
    else:
        u = None
        U0 = np.array([U0(x) for x in x])
        U = solve_numerical(x, t, U0=U0, method=method)

    if animate:
        animate_solution(x, t, U, u)

    if path is not None:
        inds = np.round(np.linspace(0, len(t)-1, time_samples)).astype(int)

        cols = [x] + ([] if u is None else [u[i,:] for i in inds]) + [U[i,:] for i in inds]
        headers = ["x "] + [f"{t[i]}" for i in inds]
        table = np.transpose(np.array(cols))
        write_table_to_file(path, table, headers)

def write_results(x, t, U, path):
    x, y = np.meshgrid(x, t)
    x, y = x.reshape(-1), y.reshape(-1)
    z = U.reshape(-1)
    print(x)
    print(y)
    print(z)
    write_table_to_file(path, np.transpose([x,y,z]), ["x t U"])

def convergence_plots(plot=False, write=False):
    runs = [
        # {"method": "crank-nicholson", "M": [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10], "N": [10, 100, 1000, 10000]}, # <- this gives O(h^2) for one or two M, but then gets unstable
        {"method": "forward-euler", "M": [5,10,15,20,25,30,35,40,45], "N": [10000, 100000, 1000000]},
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

    if plot:
        for run in runs:
            plt.title(run["method"])
            for n, N in enumerate(run["N"]):
                plt.loglog(run["M"], run["err"][n], label=f"N={N}")
            plt.show()

    if write:
        for run in runs:
            method = run["method"]
            columns = [run["M"]] + [run["err"][i] for i in range(0, len(run["N"]))]
            headers = ["M"     ] + [f"E{N}"       for N in run["N"]]

            path = f"../report/exercise4/convergence-{method}.dat"
            write_table_to_file(path, np.transpose(columns), headers)

def snapshots(plot=False, write=False):
    runs = [
        {"method": "crank-nicholson", "M": [25,50,75], "N": [10]},
        {"method": "forward-euler", "M": [20, 25, 29], "N": [100000]},
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

                if write:
                    path = f"../report/exercise4/snapshot-{method}-M{M}-N{N}.dat"
                    columns = [x, U[-1,:]]
                    headers = ["x", "U"]
                    write_table_to_file(path, np.transpose(columns), headers)

                if plot:
                    plt.plot(x, U[-1], label=f"N={N}")
                    plt.plot(x, u[-1], color="black")
            if plot:
                plt.show()

# timeevol(animate=True, path="../report/exercise4/timeevol_sin.dat", U0=lambda x: np.sin(np.pi*x), time_samples=5)
# timeevol(animate=True, time_samples=12, U0=lambda x: np.exp(-10*x**2), N=100, M=800, path="../report/exercise4/timeevol_exp.dat")

# timeevol(animate=True, M=20, N=100000, method="forward-euler")
# convergence_plots(plot=True, write=True)
# snapshots(write=True)
norm_evolution(write=True)
