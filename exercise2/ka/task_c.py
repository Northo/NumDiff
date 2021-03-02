from heateq import *


def u0(x):
    """ Initial condition """
    # return 10*np.sin(x*np.pi)
    return np.sin(x * np.pi)


bc1 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)
bc2 = BoundaryCondition(BoundaryCondition.DIRCHLET, 0)

unigrid = Grid(Grid.UNIFORM, np.linspace(0, 1, 100))
N = 100
t_end = 0.5

# Solve equation
t, U_final, sols = crank_nicolson(unigrid, bc1, bc2, u0, N, t_end)


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
# make_convergence_plots(u0, bc1, bc2)

# Animation
# animation = animate_time_development(unigrid.x, sols)
# plt.show()

# Make adaptive grid
