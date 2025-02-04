import numpy as np
import matplotlib.pyplot as plt

# Data
y1 = np.array([105, 74, 63]).T
x1 = np.array([182, 175, 170])

# Stack x with ones
X = np.vstack((x1, np.ones(len(x1)))).T
coeffs = np.array([5, 5])

# Objective function
def f(coeffs, X, y):
    y_pred = X @ coeffs
    return np.sum((y - y_pred) ** 2)

# Gradient of the objective function
def grad_f(coeffs, X, y):
    y_pred = X @ coeffs
    return -2 * X.T @ (y - y_pred)

# Gradient descent function
def gradient_descent_no_backtracking(f, grad_f, x_0, step_length=0.01, max_iter=10000, tol=1e-5):
    x = x_0
    path = [x]
    for ii in range(max_iter):
        gradient = grad_f(x, X, y1)
        if np.linalg.norm(gradient) < tol:
            break
        gradient = gradient
        x = x - step_length * gradient  # Update step
        path.append(x)
    return x, f(x, X, y1), ii, np.array(path)

# Run gradient descent
coeffs_optim, f_x, num_iter, path = gradient_descent_no_backtracking(f, grad_f, coeffs, step_length=1e-6)
print(len(path))

# Results
print("Number of iterations without backtracking: ", num_iter)
print("Final solution without backtracking: ", coeffs_optim)
print("The gradient at the final solution without backtracking: ", grad_f(coeffs_optim, X, y1))

# Pseudo-inverse solution
pseudo_inv_solution = np.linalg.pinv(X) @ y1
print("Pseudo-inverse solution: ", pseudo_inv_solution)
print("The gradient at the pseudo-inverse solution: ", grad_f(pseudo_inv_solution, X, y1))

# Contour plot for visualization
xx = np.linspace(-10, 10, 500)
yy = np.linspace(-10, 10, 500)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.zeros_like(XX)
for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        ZZ[i, j] = f([XX[i, j], YY[i, j]], X, y1)

plt.contourf(XX, YY, ZZ, levels=50, cmap="viridis")
plt.colorbar(label="Objective function value")
plt.plot(path[:, 0], path[:, 1], color="red", marker="o", label="Path")
plt.scatter(path[0, 0], path[0, 1], color="blue", label="Start")
plt.legend()
plt.xlabel("Coefficient 1")
plt.ylabel("Coefficient 2")
plt.title("Gradient Descent Path")
plt.show()
print("The gradient descent method is not able to converge to the optimal solution. The pseudo-inverse solution is the optimal solution.")
# Output the path