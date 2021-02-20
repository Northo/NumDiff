import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def make_A(n, h, order):
    a = np.ones(n-1)
    b = np.full(n, -2)
    A = np.diag(a, -1) + np.diag(a, 1) + np.diag(b, 0)
    set_h(A, h, order)
    return A

def set_h(A, h, order):
    m = len(A) - 1
    if order == 1:
        A[m, m] = -h
        A[m, m-1] = h
    elif order == 2:
        A[m, m] = -1.5*h
        A[m, m-1] = 2*h
        A[m, m-2] = -0.5*h
    return A/h**2

def make_b(x, h, alpha, sigma):
    b = f(x)
    b[0] -= alpha/h**2
    b[-1] = sigma
    return b

def solve_order_1(m, h, alpha, sigma):
    A = make_A(m+1, h, 1) 
    x = np.linspace(0, 1, m+1)
    b = make_b(x, h, alpha, sigma)
    U = lin.solve(A, b)
    return x, U

def solve_order_2(m, h, alpha, sigma):
    A = make_A(m+1, h, 2)
    x = np.linspace(0, 1, m+1)
    b = make_b(x, h, alpha, sigma)
    U = lin.solve(A, b)
    return x, U

def f(x):
    return np.cos(2*np.pi*x) + x

m = 1000
h = 0.01
alpha = 0
sigma = 1

x, U_1 = solve_order_1(m, h, alpha, sigma)
x, U_2 = solve_order_2(m, h, alpha, sigma)

plt.plot(x, U_1)
plt.plot(x, U_2)
plt.show()
