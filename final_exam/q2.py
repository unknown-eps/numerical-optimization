import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    """
    Rosenbrock function.
    :param x: Input array of shape (2,)
    :return: Function value
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def get_next_point(x_cur,base_sigma=0.01,innovation_threshold=0.2,tol=1e-4,max_retry=1000):
    """
    Generate the next point for the optimization process.
    :param x_cur: Current point
    :param base_sigma: Base standard deviation for exploration
    :param innovation_threshold: Threshold for innovation
    :return: Next point
    """
    cnt_samples = 0
    while max_retry:
        max_retry -= 1
        cnt_samples += 1
        x_next = np.random.normal(x_cur, base_sigma*(np.linalg.norm(x_cur))+tol)
        if rosenbrock(x_next) < rosenbrock(x_cur):
            accept_number = np.random.uniform(0, 1)
            if accept_number > innovation_threshold:
                return x_next,cnt_samples,max_retry
def full_optimization(x0, sigma=0.01, innovation_threshold=0.2, max_iter=1000, tol =1e-4):
    """
    Perform the full optimization process.
    :param x0: Initial point
    :param sigma: Standard deviation for exploration
    :param innovation_threshold: Threshold for innovation
    :param max_iter: Maximum number of iterations
    :return: Optimized point and function value
    """
    x_cur = x0
    path = [x_cur]
    cnt_itr= 0
    for _ in range(max_iter):
        x_next,cnt_samples,max_retry = get_next_point(x_cur, sigma, innovation_threshold)
        if max_retry == 0:
            print("Max retry reached, returning current point.")
            return np.array(path),cnt_itr
        cnt_itr += 1 + cnt_samples
        path.append(x_next)
        if rosenbrock(x_next) < tol:
            break
        x_cur = x_next
    return np.array(path),cnt_itr

np.random.seed(0)
x0 = np.array([-1.0,-1.0])
path,total_steps = full_optimization(x0, sigma=0.01, innovation_threshold=0.2, max_iter=10000, tol=1e-4)
print(f"Total points evaluated including rejected samples: {total_steps}")
print(f"Path length: {len(path)}")
print(f"Final point: {path[-1]}")
print(f"Function value at final point: {rosenbrock(path[-1])}")
# Define grid for contour plot
x_min, x_max = min(path[:, 0]) - 0.5, max(path[:, 0]) + 0.5
y_min, y_max = min(path[:, 1]) - 0.5, max(path[:, 1]) + 0.5

x_grid = np.linspace(x_min, x_max, 100)
y_grid = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Compute Rosenbrock function values on the grid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))

# Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 10), cmap='Blues', alpha=0.7)
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 10), colors='black', linewidths=0.5, alpha=0.5)
plt.colorbar(contour, label='Rosenbrock Function Value')

# Plot optimization path
plt.scatter(path[:, 0], path[:, 1], color='purple', s=15, alpha=0.6, label='Path Points')
plt.scatter(path[0, 0], path[0, 1], color='g', marker='o', s=80, label='Starting Point')
plt.scatter(path[-1, 0], path[-1, 1], color='r', marker='*', s=100, label='Starting Point')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Path on Rosenbrock Function')
plt.legend()
plt.tight_layout()
plt.savefig('optimization_path.png')
plt.show()

avg_path_length = 0
avg_total_points = 0
for test_seed in range(1,11):
    np.random.seed(test_seed)
    x0 = np.array([-1.0,-1.0])
    path,total_steps = full_optimization(x0, sigma=0.01, innovation_threshold=0.2, max_iter=10000, tol=1e-4)
    avg_path_length += len(path)
    avg_total_points += total_steps
avg_path_length /= 10
avg_total_points /= 10
print(f"Average path length over 10 runs: {avg_path_length}")
print(f"Average total points evaluated over 10 runs: {avg_total_points}")