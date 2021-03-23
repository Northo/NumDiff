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

n = 16
N = n**2

main_diag = np.full(N, -4)
off_diag = np.full(N-1, 1)
super_off_diag = np.full(N-n, 1)

# JONAS used SPARSE!
A = diags([
    main_diag, 
    off_diag,
    off_diag,
    super_off_diag,
    super_off_diag],
    (0,1,-1,n,-n)
)
#print("A: \n", A.toarray())
# THORVALD FLINCHED!

# xy contains all coordinates
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xy = np.array(np.meshgrid(x, y)) #, sparse=True)

# Initial conditions:
# Define the vector U which will contain the solution.
# Note that we use a flattened matrix (vector),
# so the index U[i,j] is accessed with index U[i*n+j].
B = np.zeros((n,n))

# The value of U_ij can be calculated from (the stencil above):
# 4*U[i,j] = U[i-1¸j] + U[i+1,j] + U[i,j-1] + U[i,j+1],
# and every index [i,j] is accessed with U[i*n+j]

# Initialize U with the boundary conditions:

def g_0(x):
    return 0

def g_1(x):
    return np.sin(2*np.pi*x)

## All values on the edges are initiated with g_0 or g_1.

# y = 0:
B[0,:n] = g_0(xy[0])

# x = 0 and x = 1:
B[:,0] = g_0(xy[:,0])
B[:,-1] = g_0(B[:,-1])

# y = 1:
B[-1,:] = g_1(xy[0,-1,:])
#print("B: \n", np.round(B, 2))

b = B.flatten()

#print(B)
#print(b)

U = np.linalg.solve(A.toarray(), b[::-1]).reshape((n,n))

print(np.round(U,2))

plt.subplot(121)
plt.title("Numerical solution")
plt.imshow(U)
plt.show()
