#import scipy.sparse as sp
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

# We want to calculate the stencil
#          U_n
#           |
#           |
#   U_w --- U   --- U_e
#           |
#           |
#          U_s

# Which can be written as a (useless, but it took time 
# to derive so I'm not deleting it) matrix equation
# AU + BU^T = 0,
# where U is our approximated solution matrix of size n*n, and
# A and B are tridiagonal matrices with (1,-4,1) and (1,0,1)
# as their diagonals. ^T denotes transposition.

# This equation is not very practical, however. 
# It's easier to flatten it and use one matrix 
# of size N*N, where N = n*n.

Nx = 100
hx = 1/(1+Nx)
Ny = 100
hy = 1/(1+Ny)

main_diag = np.full(Nx*Ny, -2/hx**2 - 2/hy**2)
off_diag = np.full(Nx*Ny-1, 1/hy**2)
super_off_diag = np.full(Ny*(Nx-1), 1/hx**2)

A = csr_matrix(diags([
    main_diag, 
    off_diag,
    off_diag,
    super_off_diag,
    super_off_diag],
    (0,1,-1,Nx,-Nx)
))

# xy contains all coordinates
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
xy = np.meshgrid(x, y, sparse=True)

# Define boundary functions:
def g_0(x):
    return 0

def g_1(x):
    return np.sin(2*np.pi*x)

# Initialize system with boundary conditions:
# Define a vector b which holds the right 
# hand side of the equation A*u = b.
# Initializing b with a 2d matrix B, to keep the code intuitive.
# After flatteni1g B, we get a matrix where the
# element B[i,j] is accessed with index B[i*n+j].
b = np.zeros(Ny*Nx)

# All values on the edges are initiated with g_0 or g_1.
# y = 1:
b[0:Nx] = g_1(x)/hy**2
# x = 0 and x = 1:
b[0::Nx] = g_0(y)
b[Nx-1::Nx] = g_0(y)
# y = 0:
b[Nx*(Ny-1):] = g_0(x)
#b = B.flatten()
print(b)

# The value of U_ij ise calculated from (the stencil above):
# 4*U[i,j] = U[i-1¸j] + U[i+1,j] + U[i,j-1] + U[i,j+1],
# where every index [i,j] is accessed with U[i*n+j]
# This means that we will solve the equation 
# Au = b.
U = spsolve(A, b)
U = U.reshape((Ny,Nx))

plt.figure()
plt.title("Numerical solution")
# Flip U because in our original array, x = 0 lies at the top.
plt.imshow(np.flip(U, axis=1), extent=[0,1,0,1])
plt.show()
