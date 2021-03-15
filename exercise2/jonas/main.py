import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation

from solvers import *
from plotters import *

def f(x):
    return 2*np.pi*x - np.sin(2*np.pi*x)

def g(x):
    return 1-np.abs(2*x-1)

def burger(x):
    return np.exp(-400*(x - 0.5)**2)

def g_0_d(f, t):
    return f(0)

def g_1_d(f, t):
    return f(1)

def g_0_n(f, t):
    return 0

def g_1_n(f, t):
    return 0

def main_jonas():
    m = 100
    N = 100
    t_end = 1

    # The following illustrates that Backward Euler breaks down at
    # some M or N. We suspect the ratio M/N determines this.
    #x_be, U_be = backward_euler(f, 39, N, t_end, g_0_n, g_1_n, bc="n")
    #plt.plot(x_be, U_be[40])
    #plt.show()
    #x_be, U_be = backward_euler(f, 42, N, t_end, g_0_n, g_1_n, bc="n")
    #plt.plot(x_be, U_be[40])
    #plt.show()
    #x_be, U_be = backward_euler(f, 59, N, t_end, g_0_n, g_1_n, bc="n")
    #plt.plot(x_be, U_be[40])
    #plt.show()

    x_be, U_be = backward_euler(f, m, N, t_end, g_0_d, g_1_d, bc="d")
    x_cn, U_cn = crank_nicolson(f, m, N, t_end, g_0_d, g_1_d, bc="d")

    anim_be = animate_time_development(x_be, U_be, "Soltuion with backward Euler, Dirichlet conditions")
    plt.show()

    anim_cn = animate_time_development(x_cn, U_cn, "Soltuion with Crank-Nicolson, Dirichlet conditions")
    plt.show()
        
    x_be, U_be = backward_euler(f, 44, N, t_end, g_0_n, g_1_n, bc="n")
    x_cn, U_cn = crank_nicolson(f, m, N, t_end, g_0_n, g_1_n, bc="n")


    anim_be = animate_time_development(x_be, U_be, "Soltuion with backward Euler, Neumann conditions")
    plt.show()

    anim_cn = animate_time_development(x_cn, U_cn, "Soltuion with Crank-Nicolson, Neumann conditions")
    plt.show()
        
def convergence():
    m = 50
    N = 50 
    t_end = 1

    convergence_plot(backward_euler, f, m, N, t_end, g_0_d, g_1_d, bc="d")
    convergence_plot(crank_nicolson, f, m, N, t_end, g_0_d, g_1_d, bc="d")
    convergence_plot(backward_euler, f, m, N, t_end, g_0_n, g_1_n, bc="n")
    convergence_plot(crank_nicolson, f, m, N, t_end, g_0_n, g_1_n, bc="n")

#main_jonas()
convergence()
