import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation


# Shamelessly stolen from KA
def piecewise_constant_continuation(xr, ur):
    """
    make a piecewise constant function of spacial coordinate x from a reference solution u

    Parameters:
        xr : x grid for the reference solution
        ur : Array, the reference solution
    Returns:
        numpy.piecewise function, piecewise constant funciton of x
    """

    return lambda x: np.piecewise(
        x,
        [xr[i] <= x < xr[j] for (i, j) in zip(range(len(ur) - 1), range(1, len(ur)))],
        ur,
    )

def pc(xref, uref):
    return lambda x: np.piecewise(x, [xref[i] <= x <= xref[j] for (i,j) in zip(range(len(uref)-1), range(1,len(uref)))], uref)


def convergence_plot(solver, f, m, N, t_end, g_0_d, g_1_d, bc):
    M = 3*m
    t = 10

    x_ref, ref_sol = solver(f, M, N, t_end, g_0_d, g_1_d, bc=bc)
    ref_piecewise = np.vectorize(pc(x_ref, ref_sol[t,:]))

    M_array = [8,16,32,64,128,256,512,1024]
    e = np.zeros(len(M_array))
    for (i, m_i) in enumerate(M_array):
        x_sol, U = solver(f, m_i, N, t_end, g_0_d, g_1_d, bc=bc)
        U_ref = ref_piecewise(x_sol)
        e[i] = error(U_ref, U[t,:])
    
    plt.plot(M_array, e, marker="o", linestyle="--")
    plt.yscale("log")
    plt.xscale("log", basex=2)
    plt.ylabel("M")
    plt.ylabel("Error")
    plt.title(str(solver))
    plt.show()


def error(ref_sol, sol):
    '''
        Returns the error of a computated solution at a specific time.
        
        Arguments:
            ref_sol:    Reference solution, wtih a large amount of spatial steps.
            sol:        Solutions to be evaluated.
            t:          Time index for which evaluation happens.
        Returns:
            e:          Error (scalar) defined in project description.
    '''

    m = len(sol)
    step = int(len(ref_sol)/m)  # The step length when iterating through ref_sol.
                                # We must iterate through ref_sol so that
                                # we only use the indices corresponding to
                                # indices in sol, so that we evaluate the
                                # two solutions in the same values of x.

    #ref_slice = ref_sol[t,::step]
    #sol_slice = sol[t,:]
    a = np.sum((ref_sol - sol)**2)/(m+2)
    b = np.sum(ref_sol**2)/(m+2)
    return np.sqrt(a/b)


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

