import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functional_form import (
    augment_data,
    batch_gradient_descent,
    mini_batch_gradient_descent,
    mini_batch_adam,
)


# Function to get the value of eta for a given x and parameters
def get_eta(x, params):
    x = augment_data(x.reshape(1, -1)).reshape(-1)
    return np.dot(x, params)


# Function to get the gradient of eta for a given x and parameters
def grad_eta(x, params):
    grad = params[1:6].copy()
    start_pos = 6
    for i in range(5):
        for j in range(i, 5):
            grad[i] += params[start_pos] * x[j]
            grad[j] += params[start_pos] * x[i]
            start_pos += 1
    return grad


# Function to get the hessian of eta for a given x and parameters
def hessian_eta(x, params):
    hessian = np.zeros((5, 5))
    start_pos = 6
    for i in range(5):
        for j in range(i, 5):
            hessian[i, j] += params[start_pos]
            hessian[j, i] += params[start_pos]
            start_pos += 1
    return hessian


# Function to get the error for a given x and parameters
def get_error(x, params):
    return -get_eta(x, params)


# Function to get the gradient of error for a given x and parameters
def grad_error(x, params):
    return -grad_eta(x, params)


# Function to get the hessian of error for a given x and parameters
def hessian_error(x, params):
    return -hessian_eta(x, params)


## Optimization methods


# Function to perform newton's method


def newton_method(x0, max_iter=1000, tol=1e-6):
    """
    Each optimization method only return the path of the optimization as a list of numpy arrays
    """
    path = [x0.copy()]
    x = x0.copy()
    for iter_cnt in range(max_iter):
        grad = grad_error(x, params)
        if np.linalg.norm(grad) < tol:
            break
        hessian = hessian_error(x, params)
        x = x - np.linalg.inv(hessian) @ grad
        path.append(x)
    return path


def bfgs(x0, max_iter=5, tol=1e-6, c1=1e-4, r=0.5):
    path = [x0.copy()]
    x = x0.copy()
    hess_approx = np.eye(5)
    grad = grad_error(x, params)

    for iter_cnt in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        dir_descent = -hess_approx @ grad
        alpha = 1
        while get_error(x + alpha * dir_descent, params) > get_error(
            x, params
        ) + c1 * alpha * np.dot(grad, dir_descent):
            alpha *= r
        x_new = x + alpha * dir_descent
        grad_new = grad_error(x_new, params)
        s = x_new - x
        y = grad_new - grad
        if np.dot(s, y) < 1e-8:
            hess_approx = np.eye(5)
        else:
            pho = 1 / np.dot(s, y)
            hess_approx = (np.eye(5) - pho * np.outer(s, y)) @ hess_approx @ (
                np.eye(5) - pho * np.outer(y, s)
            ) + pho * np.outer(s, s)
        x = x_new
        grad = grad_new
        path.append(x)
    return path


def conjugate_gradient_method(x0, max_iter=1000, tol=1e-6):
    path = [x0.copy()]
    x = x0.copy()
    r = grad_error(x, params)
    p = -r
    for iter_cnt in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        alpha = -np.dot(r, p) / np.dot(p, hessian_error(x, params) @ p)
        x = x + alpha * p
        path.append(x)
        r = grad_error(x, params)
        beta = np.dot(r, hessian_error(x, params) @ p) / np.dot(
            p, hessian_error(x, params) @ p
        )
        p = -r + beta * p
    return path


# Plotting function
def make_plots(path, params, method_name, index_list=None):
    """
    Index:List of length 2 if None coordinates with highest variance in the path are taken
    Plots contour for the coordinates at index_list[0],index_list[1] and the path of the optimization
    """
    path = np.vstack(path)
    eta_list = [get_eta(x, params) for x in path]
    plt.plot(eta_list, label="Value of eta", c="blue", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Value of eta")
    plt.title(f"Value of eta vs iteration for {method_name}")
    plt.savefig(f"./part2_plots/{method_name}_eta.png")
    plt.show()

    ## Take coordinates with highest variance if index_list is None
    if index_list is None:
        index_list = np.argsort(np.var(path, axis=0))[-2:]
    fig, ax = plt.subplots()
    min_x1 = min(path[:, index_list[0]]) - 1
    max_x1 = max(path[:, index_list[0]]) + 1
    min_x2 = min(path[:, index_list[1]]) - 1
    max_x2 = max(path[:, index_list[1]]) + 1
    x1 = np.linspace(min_x1, max_x1, 100)
    x2 = np.linspace(min_x2, max_x2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1).flatten()
    for i in range(len(Z)):
        temp_x = path[-1].copy()
        temp_x[index_list[0]] = X1.flatten()[i]
        temp_x[index_list[1]] = X2.flatten()[i]
        Z[i] = get_eta(temp_x, params)
    Z = Z.reshape(X1.shape)
    plt.contourf(X1, X2, Z, 20, alpha=0.8, cmap="viridis")
    plt.colorbar()
    plt.plot(
        path[:, index_list[0]],
        path[:, index_list[1]],
        label="Path of optimization",
        marker="o",
        color="black",
    )
    plt.xlabel(f"Value of x{index_list[0]}")
    plt.ylabel(f"Value of x{index_list[1]}")
    plt.title(f"Contour plot for {method_name}")
    plt.scatter(
        path[0, index_list[0]],
        path[0, index_list[1]],
        c="green",
        label="Initial point",
        s=100,
    )
    plt.scatter(
        path[-1, index_list[0]],
        path[-1, index_list[1]],
        c="red",
        label="Final point",
        s=100,
    )
    plt.legend()
    plt.savefig(f"./part2_plots/{method_name}_contour.png")
    plt.show()


###
df = pd.read_csv("mid_term_data.csv")
X_without_aug = df.drop("eta", axis=1).values
scaler = StandardScaler()
X_without_aug = scaler.fit_transform(X_without_aug)
X_aug = augment_data(X_without_aug)
y = df["eta"].values

### Taking use input for the type of gradient descent
user_choice = input(
    "Enter 1 for  Batch Gradient Descent, 2 for Mini Batch gradient, 3 for Mini batch Adam: "
)

num_epochs = 10000
batch_size = 100
base_name = None
if user_choice == "1":
    _, params, _, error_list = batch_gradient_descent(X_aug, y, num_epochs=num_epochs)
    base_name = "batch gradient descent"

elif user_choice == "2":
    _, params, _, error_list = mini_batch_gradient_descent(
        X_aug, y, num_epochs=num_epochs, batch_size=batch_size
    )
    base_name = "mini batch gradient descent"
elif user_choice == "3":
    _, params, _, error_list = mini_batch_adam(
        X_aug, y, num_epochs=num_epochs, batch_size=batch_size
    )
    base_name = "mini batch adam"

print(f"Parameters are: {params}")
print(f"Final error is {error_list[-1]}")

## From now on x refers to the parameters we want to optimize over and not the data

x0 = np.zeros(5)  # Initial guess
print("Initial value of eta is", get_eta(x0, params))

## Newton method
path = newton_method(x0)
print("Prinitin for newton method")
print("final value of x in is", path[-1])
print("Unscaled final value of x is", scaler.inverse_transform(path[-1].reshape(1, -1)))
print("Final value of eta is", get_eta(path[-1], params))
print("Number of iterations taken by newton method is", len(path) - 1)
print("\n")
make_plots(path, params, base_name + " with Newton's method")


## BFGS
path = bfgs(x0)
print("Priniting for BFGS method")
print("final value of x in is", path[-1])
print("Unscaled final value of x is", scaler.inverse_transform(path[-1].reshape(1, -1)))
print("Final value of eta is", get_eta(path[-1], params))
print("Number of iterations taken by BFGS method is", len(path) - 1)
print("\n")
make_plots(path, params, base_name + " with BFGS method")
## Conjugate gradient
path = conjugate_gradient_method(x0)
print("Priniting for conjugate gradient method")
print("final value of x in is", path[-1])
print("Unscaled final value of x is", scaler.inverse_transform(path[-1].reshape(1, -1)))
print("Final value of eta is", get_eta(path[-1], params))
print("Number of iterations taken by conjugate gradient method is", len(path) - 1)
make_plots(path, params, base_name + " with Conjugate gradient method")
