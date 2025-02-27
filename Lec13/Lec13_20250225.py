import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    return 4 * x[0] ** 2 + x[1] ** 2


x = np.linspace(-4, 4, 50)
y = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x, y)
Z = f1([X, Y])
fig = plt.figure()
plt.contourf(X, Y, Z, levels=np.linspace(0, 5, 50), cmap="Blues")
plt.savefig("f1.png")
plt.show()


def exact_line_search(f, x, d, alpha_inc=1e-3):
    alpha = 0.0
    while f(x + (alpha + alpha_inc) * d) < f(x + alpha * d):
        alpha = alpha + alpha_inc
    if alpha == 0.0:
        while f(x + (alpha - alpha_inc) * d) < f(x + alpha * d):
            alpha = alpha - alpha_inc
    return alpha


def coordinate_descent(f, x0, tol=1e-6):
    path = [x0]
    x = x0
    n = x0.shape[0]
    for i in range(n):
        dir_search = np.zeros(n)
        dir_search[i] = 1
        alpha = exact_line_search(f, x, dir_search, alpha_inc=tol)
        x = x + alpha * dir_search
        path.append(x)
    return x, path


x0 = np.array([-1, -1])
x_opt, path = coordinate_descent(f1, x0)
path = np.array(path)
print("Optimal solution", x_opt)
fig = plt.figure()
plt.contourf(X, Y, Z, levels=np.linspace(0, 10, 50), cmap="Blues")
plt.colorbar()
plt.plot(path[:, 0], path[:, 1], "ro-")
plt.savefig("f1 path.png")
plt.show()


def f2(x):
    return 4*x[0]**2+x[1]**2-2*x[0]*x[1]

x_opt,path=coordinate_descent(f2,x0)
print(f'Optimal soln f2:{x_opt}')
fig=plt.figure()
plt.contourf(X,Y,f2([X,Y]),levels=np.linspace(0,10,50),cmap='Blues')
plt.colorbar()
path=np.array(path)
plt.plot(path[:,0],path[:,1],'ro-')
plt.savefig('f2 path.png')
plt.show()


