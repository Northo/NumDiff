from heateq import *

# Is this a good manufactured solution/bc/initial cond? :p

PLOT_SAMPLES = False


def u0(x):
    """ Initial condition """
    # return 10*np.sin(x*np.pi)
    return np.sin(x * np.pi)


# Setup
bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
unigrid = Grid(Grid.UNIFORM, np.linspace(0, 1, 100))
N = 100
t_end = 0.5

# Solve equation
t, U_final, sols = crank_nicolson(unigrid, bc1, bc2, u0, N, t_end)

if PLOT_SAMPLES:
    n_samples = 5
    for i in range(n_samples):
        j = i * N // n_samples
        ti = t[j]
        plt.plot(unigrid.x, sols[j], ".", label=f"t={ti}")
    plt.show()


def make_convergence_plots(u0, bc1, bc2):
    M_ref = 1000
    N = 100
    ref_grid = Grid(Grid.UNIFORM, np.linspace(0, 1, M_ref))
    discrete_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 1, plot=True
    )
    discrete_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 1, plot=True
    )

    continous_convergence_plot(
        backward_euler, ref_grid, bc1, bc2, u0, 200, N, 1, plot=True
    )
    continous_convergence_plot(
        crank_nicolson, ref_grid, bc1, bc2, u0, 200, N, 1, plot=True
    )


# Convergence plots (shabby yea)
make_convergence_plots(u0, bc1, bc2)

# Animation
# animation = animate_time_development(unigrid.x, sols)
# plt.show()

# Make adaptive grid


def error_func(a, c, b, u0=u0):
    """ Estimate of meassure of error of some sort or something """
    return np.abs(u0(c) - (u0(a) + u0(b)) / 2)


x_adapt = partition_interval(0, 1, error_func, 0.001)


adapt_grid = Grid(Grid.NON_UNIFORM, x_adapt)
t, U_final, sols = crank_nicolson(adapt_grid, bc1, bc2, u0, N, t_end)

plt.plot(x_adapt, np.zeros(len(x_adapt)), ".")
plt.plot(x_adapt, U_final)
plt.title(f"Adaptive grid, # points: {len(x_adapt)}")
plt.show()
