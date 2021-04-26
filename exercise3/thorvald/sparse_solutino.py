import scipy.sparse as sp
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

def anal(x, y):
    return np.sin(2 * np.pi * x) * np.sinh(2 * np.pi * y) / np.sinh(2 * np.pi)

def get_K(N):
    return diags(
        [np.full(N, -2), np.ones(N-1), np.ones(N-1)],
        [0, 1, -1],
    )

def get_A(Nx, Ny):
    hx = 1/(Nx + 2 - 1)
    hy = 1/(Ny + 2 - 1)
    Kx = get_K(Nx)
    Ky = get_K(Ny)
    # A = (
    #     sp.kron(Ky, sp.eye(Nx))/hx**2
    #     + sp.kron(sp.eye(Ny), Kx)/hy**2
    # )
    A = (
        sp.kron(sp.eye(Ny), Kx)/hx**2
        + sp.kron(Ky, sp.eye(Nx))/hy**2
    )
    return A

# Define boundary functions:
def g_0(x):
    return 0


def g_1(x):
    return np.sin(2*np.pi*x)


def get_b(Nx, Ny):
    # Initialize system with boundary conditions:
    # Define a vector b which holds the right 
    # hand side of the equation A*u = b.
    # Initializing b with a 2d matrix B, to keep the code intuitive.
    # After flatteni1g B, we get a matrix where the
    # element B[i,j] is accessed with index B[i*n+j].
    b = np.zeros(Ny*Nx)

    # All values on the edges are initiated with g_0 or g_1.
    x = np.linspace(0, 1, Nx+2)[1:-1]
    y = np.linspace(0, 1, Ny+2)[1:-1]
    hx = 1/(Nx + 2 - 1)
    hy = 1/(Ny + 2 - 1)
    b[-Nx:] = -g_1(x)/hy**2
    return b


def get_solution(Nx, Ny):
    """Assumes default BCs"""
    A = get_A(Nx, Ny)
    b = get_b(Nx, Ny)
    U = spsolve(A, b)
    return U


def solve_errors(Nxs, Nys):
    assert len(Nxs) == len(Nys)
    errors = []
    def err_func(approx, anal):
        approx = approx.flatten()
        anal = anal.flatten()
        order = 2
        return (
            np.linalg.norm(approx - anal, ord=order) /
            np.linalg.norm(anal, ord=order)
        )
    for Nx, Ny in zip(Nxs, Nys):
        x = np.linspace(0, 1, Nx+2)[1:-1]
        y = np.linspace(0, 1, Ny+2)[1:-1]
        xx, yy = np.meshgrid(x, y)
        u = anal(xx, yy)
        U = get_solution(Nx, Ny)
        errors.append(err_func(U, u))
    return errors

def write_to_file(path, U, x, y):
    n = len(x)
    m = len(y)
    assert(len(U) == n*m)
    
    # Create an array of size (3,n*m), so that we get something like
    #####################
    # x     y       U
    # 0     0       U_00
    # 0.1   0       U_10
    # ...   ...     ...
    # 0     0.1     U_01
    # ...   ...     ...
    #####################

    data = np.array([np.tile(x,m), np.sort(np.tile(y,n)), U])
    print(data.shape)
    headers = "x y U"
    print(np.transpose(data).shape)
    np.savetxt(path, np.transpose(data), header=headers, comments="", fmt="%1.3f")
    print(f"Wrote to {path}")


def write_table_to_file(path, table, headers):
    np.savetxt(path, table, header=" ".join(headers), comments="")
    print(f"Wrote to {path}")

def write_columns_to_file(path, columns, headers):
    max_length = np.max([len(column) for column in columns])
    for i in range(0, len(columns)):
        length = len(columns[i])
        column = np.full(max_length, np.nan)
        column[0:length] = columns[i]
        columns[i] = column
    write_table_to_file(path, np.transpose(columns), headers)

def plot_stencil(A):
    plt.imshow(A)
    plt.show()

def save_stencil(A, filename):
    headers = "x y U"
    N = np.prod(A.shape)
    data = np.empty((N, 3))
    for i, coord in enumerate(np.ndindex(A.shape)):
        data[i] = np.array([*coord, A[coord]])
    np.savetxt(filename, data, header=headers, comments="", fmt="%1.3f")


def discretization_test():
    N0 = 100
    num_points = 12
    var_low = 0.9
    var_high = 0.99
    N_var = np.geomspace(N0*(1-var_high), N0*(10), num_points).astype(int)

    def save(filename, errors, nxs, nys):
        data = np.stack([nxs*nys, nxs, nys, errors]).T
        header = "N nx ny error"
        np.savetxt(
            filename,
            data,
            header=header,
            comments='',
        )
    ## Change Ny
    Nx = N0
    Nxs = [Nx] * num_points
    Nys = N_var
    #Nys = (
        #N0 * ( 1 + var * np.linspace(-1, 1, num_points))
    #).astype(int)
    save(f"../../report/exercise3/convergence/varNY.dat", solve_errors(Nxs, Nys), Nxs, Nys)
    #save(f"../../report/exercise3/convergence/varNY-nx-{Nx}-num-{num_points}-var-{var}.dat", solve_errors(Nxs, Nys), Nxs, Nys)

    ## Change Nx
    Ny = N0
    Nys = [Ny] * num_points
    Nxs = N_var
    #Nxs = (
    #    N0 * ( 1 + var * np.linspace(-1, 1, num_points))
    #).astype(int)
    save(f"../../report/exercise3/convergence/varNX.dat", solve_errors(Nxs, Nys), Nxs, Nys)
    #save(f"../../report/exercise3/convergence/varNX-ny-{Ny}-num-{num_points}-var-{var}.dat", solve_errors(Nxs, Nys), Nxs, Nys)

    ## Change both
    Nxs = N_var[4:num_points-2]
    Nys = N_var[4:num_points-2]
    #Nys = Nxs = (
    #    N0 * ( 1 + 0.5 * var * np.linspace(-1, 1, num_points))
    #).astype(int)
    save(f"../../report/exercise3/convergence/both.dat", solve_errors(Nxs, Nys), Nxs, Nys)
    #save(f"../../report/exercise3/convergence/both-{Ny}-num-{num_points}-var-{var}.dat", solve_errors(Nxs, Nys), Nxs, Nys)
discretization_test()
