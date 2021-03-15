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

## Set up the problem ##
# We need to solve for all the internal points,
# so a My x Mx sized problem.
#
# We have a grid, width Mx and height My.
# In each point on the grid, we have a value
# U(x, y).
# We think of this of course as a 2D array.
# Imagine now that we flatten this array, giving
# us a 1D array, where the first Mx elements
# are the first row, the next Mx elements the
# second row, and so on.
# We construct now the five point stencil
# matrix for this array.
#
#        u_top
#          |
# u_left - u - u_right
#          |
#      u_bottom
#
# Right and left are simple, as they in the
# flattened array are simply the next and
# previous elements.
# Up and down are also quite simple, as they
# are the elements Mx before and after.

diagonal = np.full(Mx * My, -2/hx**2 -2/hy**2)
off_diagonal = np.full(Mx * My - 1, 1/hx**2)
super_off_diagonal = np.full(Mx * (My - 1), 1/hy**2)

A = (
    np.diag(diagonal)
    + np.diag(off_diagonal, k=1)
    + np.diag(off_diagonal, k=-1)
    + np.diag(super_off_diagonal, k=Mx)
    + np.diag(super_off_diagonal, k=-Mx)
)

# We must now consider the boundary conditions
bc = np.zeros(Mx * My)
bc[0:Mx] = -g_x1(np.linspace(0, 1, Mx+2)[1:-1])/hy**2

u = np.linalg.solve(A, bc)

# Reshape and fit our solution on a grid
U = np.zeros((Mx+2, My+2))
U[0, :] = g_x1(np.linspace(0, 1, Mx+2))
U[1:-1, 1:-1] = u.reshape((Mx, My))

plt.subplot(121)
plt.title("Numerical solution")
plt.imshow(U)


plt.subplot(122)
plt.title("Thorvad's anal. solution")
# Thorvald's analytical solution
x = np.linspace(0, 1, Mx)
y = np.linspace(0, 1, My)
xx, yy = np.meshgrid(x, y)
plt.pcolormesh(xx, yy, np.sinh(2*np.pi*yy) * np.sin(2*np.pi*xx) / np.sinh(2*np.pi))

plt.tight_layout()
plt.show()
