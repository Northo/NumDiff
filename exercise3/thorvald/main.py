import numpy as np
import matplotlib.pyplot as plt

## Boundary conditions ##
g_0y = 0
g_x0 = 0
g_1y = 0
g_x1 = lambda x: np.sin(2*np.pi*x)

## The grid ##
# Mx and My are internal points in x and y direction.
Mx = 100
My = 100
hx = 1 / (Mx + 1)
hy = 1 / (My + 1)

u = np.zeros((Mx+2, My+2))
u[0, :] = g_0y
u[:, 0] = g_x0
u[1, :] = g_1y
u[:, 1] = g_x1(np.linspace(0, 1, Mx+1))

## Five point stencil ##
u_right = np.roll(u, -1, axis=1)
u_left = np.roll(u, 1, axis=1)
u_up = np.roll(u, 1, axis=0)
np_down = np.roll(u, -1, axis=0)

# Take each direction separately
u_yy = (u_up + u_down - 2*u) / hy**2
u_xx = (u_right + u_left -2*u) / hx**2
u_dd = u_yy + u_xx

# Solve the problem
