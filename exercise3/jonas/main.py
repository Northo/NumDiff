from scipy.sparse import diags
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

Nx = 30
hx = 1/(1+Nx)
Ny = 17
hy = 1/(1+Ny)

main_diag = np.full(Nx*Ny, -2/hx**2 - 2/hy**2)
off_diag = np.full(Nx*Ny-1, 1/hy**2)
super_off_diag = np.full(Ny*(Nx-1), 1/hx**2)

# JONAS used SPARSE!
A = diags([
    main_diag, 
    off_diag,
    off_diag,
    super_off_diag,
    super_off_diag],
    (0,1,-1,Nx,-Nx)
)
#print("A: \n", A.toarray())
# THORVALD FLINCHED!

# xy contains all coordinates
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
xy = np.array(np.meshgrid(x, y)) #, sparse=True)

# Initial conditions:
# Define the vector U which will contain the solution.
# Note that we use a flattened matrix (vector),
# so the index U[i,j] is accessed with index U[i*n+j].
B = np.zeros((Ny, Nx))
print(B)

# The value of U_ij can be calculated from (the stencil above):
# 4*U[i,j] = U[i-1¸j] + U[i+1,j] + U[i,j-1] + U[i,j+1],
# and every index [i,j] is accessed with U[i*n+j]

# Initialize U with the boundary conditions:

def g_0(x):
    return 0

def g_1(x):
    return np.sin(2*np.pi*x)

## All values on the edges are initiated with g_0 or g_1.
# Using a 2d matrix here, to keep the code intuitive.
# TODO: Use sparse matrix instead.
# y = 0:
B[0,:Nx] = g_0(xy[0])

# x = 0 and x = 1:
B[:,0] = g_0(xy[:,0])
B[:,-1] = g_0(B[:,-1])

# y = 1:
B[-1,:] = g_1(xy[0,-1,:])/hx**2
#print("B: \n", np.round(B, 2))

b = B.flatten()

#print(B)
#print(b)

U = np.linalg.solve(A.toarray(), b[::-1]).reshape((Ny,Nx))

print(np.round(U,2))

plt.subplot(121)
plt.title("Numerical solution")
plt.imshow(U)
plt.show()
