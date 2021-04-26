#!/usr/bin/sage

m, n = var("m n")
h, k = var("h k")
c(m,n,h,k) = h^m*k^n/(factorial(m)*factorial(n)) # coeff that multiplies del_x^m del_y^n when taylor expanding u(x+h,y+k)

stencil5 = [
	[0,  1,  0],
	[1, -4,  1],
	[0,  1,  0]
]
f5(m,n) = -4*c(m,n,0,0) + c(m,n,0,+h) + c(m,n,0,-h) + c(m,n,+h,0) + c(m,n,-h,0)
f9(m,n) = -10/3 * c(m,n,0,0) + 2/3 * (c(m,n,0,+h) + c(m,n,0,-h) + c(m,n,+h,0) + c(m,n,-h,0)) + 1/6 * (c(m,n,+h,+h) + c(m,n,+h,-h) + c(m,n,-h,+h) + c(m,n,-h,-h))
for f, order in zip((f5, f9), (4, 6)):
	print(f)
	for m in range(0, order+1):
		for n in range(0, order-m+1):
			if f(m,n) != 0:
				print(m, n, f(m,n)/h^2)
