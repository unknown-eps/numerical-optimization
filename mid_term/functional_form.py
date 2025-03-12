import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def error_func(X, y, params):
    """
    Ensure that params is of the shape X.shape[1],None or X.shape[1],k
    A vector of dim 1 or k is returned corresponding to the input shapes
    """
    if len(params.shape) == 1:
        return (X @ params - y).T @ (X @ params - y) / X.shape[0]
    else:
        return (X @ params - y.reshape(X.shape[0], 1)).mean(
            axis=0
        )  # Using least squares error axis=0 is added to handle when we are plotting for multiple params


def error_grad(X, y, params):
    """
    Returns the gradient of the error function with respect to the parameters
    This expects params to be of shape(X.shape[1]) y to be of shape X.shape[0]
    """
    return (
        2 * (X.T) @ (X @ params - y) / X.shape[0]
    )  # Output is of the shape X.shape[1]


def get_batch(X_aug, y, batch_size, start_idx):
    """
    Returns the batch of data starting from start_idx and ending at start_idx+batch_size
    """
    end_idx = min(X_aug.shape[0], start_idx + batch_size)  # end_idx is not included
    return X_aug[start_idx:end_idx, :], y[start_idx:end_idx]


def augment_data(X_without_aug):
    """
    Augments the data with degree 2 terms and adds a bias term
    """
    n_samples = X_without_aug.shape[0]
    final_feature_list = [np.ones((n_samples, 1))]
    for idx, f1 in enumerate(
        X_without_aug.T
    ):  # as features are stored as columns f1 is of the shape N_samples
        final_feature_list.append(f1.reshape(n_samples, -1))
        for f2 in X_without_aug.T[idx:, :]:
            final_feature_list.append(
                (f1 * f2).reshape(n_samples, -1)
            )  # reshape is needed as lated I am going to h stack
    return np.hstack(final_feature_list)


def batch_gradient_descent(
    X_aug,
    y,
    params0=None,
    lr=1e-2,
    num_epochs=100,
    tol=1e-6,
):
    """
    Return number of iterations the final params and a list of params at each iteration.
    Stopping condition norm of the gradient is less than tol
    """
    params0 = params0 if params0 is not None else np.zeros(X_aug.shape[1])
    path_list = []
    path_list.append(params0.copy())
    cur_params = params0
    error_list = [error_func(X_aug, y, cur_params)]
    for iter_count in range(num_epochs):
        grad = error_grad(X_aug, y, cur_params)
        if np.linalg.norm(grad) < tol:
            break
        cur_params = cur_params - lr * grad
        error_list.append(error_func(X_aug, y, cur_params))
        path_list.append(cur_params.copy())
    return iter_count + 1, cur_params, path_list, error_list


def mini_batch_gradient_descent(
    X_aug,
    y,
    params0=None,
    lr=1e-2,
    num_epochs=100,
    tol=1e-6,
    batch_size=100,
):
    """
    Return number of iterations the final params and a list of params at each iteration.
    Stopping condition norm of the gradient is less than tol
    """
    params0 = params0 if params0 is not None else np.zeros(X_aug.shape[1])
    path_list = []
    path_list.append(params0.copy())
    cur_params = params0
    error_list = [error_func(X_aug, y, cur_params)]
    for iter_count in range(num_epochs):
        grad_avg = 0
        for start_idx in range(0, X_aug.shape[0], batch_size):
            X_batch, y_batch = get_batch(X_aug, y, batch_size, start_idx)
            grad = error_grad(X_batch, y_batch, cur_params)
            grad_avg += grad * X_batch.shape[0]
            cur_params = cur_params - lr * grad
            path_list.append(cur_params.copy())
            error_list.append(error_func(X_aug, y, cur_params))
        if np.linalg.norm(grad_avg) / X_aug.shape[0] < tol:
            break
    return iter_count + 1, cur_params, path_list, error_list


def mini_batch_adam(
    X_aug,
    y,
    params0=None,
    lr=1e-2,
    num_epochs=100,
    tol=1e-6,
    batch_size=100,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    Return number of iterations, the final params, a list of params at each iteration,error at each iteration.
    Stopping condition norm of the gradient is less than tol
    """
    params0 = params0 if params0 is not None else np.zeros(X_aug.shape[1])
    path_list = []
    path_list.append(params0.copy())
    cur_params = params0
    m = np.zeros_like(params0)
    v = np.zeros_like(params0)
    error_list = [error_func(X_aug, y, cur_params)]
    for iter_count in range(num_epochs):
        grad_avg = 0
        for start_idx in range(0, X_aug.shape[0], batch_size):
            X_batch, y_batch = get_batch(X_aug, y, batch_size, start_idx)
            grad = error_grad(X_batch, y_batch, cur_params)
            grad_avg += grad * X_batch.shape[0]
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1 ** (iter_count + 1))
            v_hat = v / (1 - beta2 ** (iter_count + 1))
            cur_params = cur_params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            path_list.append(cur_params.copy())
            error_list.append(error_func(X_aug, y, cur_params))
        if np.linalg.norm(grad_avg) / X_aug.shape[0] < tol:
            break
    return iter_count + 1, cur_params, path_list, error_list


def visualize_path(path_list, index_1, index_2, algo_name, batch_size=None):
    """
    Index 1 and Index 2 represent the index of the respective parameters in the path_list
    """
    path_list = np.vstack(path_list)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the meshgrid for the plot
    v1, v2 = np.meshgrid(
        np.linspace(
            np.min(path_list[:, index_1]) - 1, np.max(path_list[:, index_1]) + 1, 100
        ),
        np.linspace(
            np.min(path_list[:, index_2]) - 1, np.max(path_list[:, index_2]) + 1, 100
        ),
    )

    # Calculate the error for each combination of parameters
    error = np.zeros_like(v1)
    for i in range(v1.shape[0]):
        for j in range(v1.shape[1]):
            params = path_list[-1].copy()
            params[index_1] = v1[i, j]
            params[index_2] = v2[i, j]
            error[i, j] = error_func(X_aug, y, params)

    # Plot the contour
    contour = ax.contourf(v1, v2, error, levels=50, cmap="coolwarm")
    fig.colorbar(contour, ax=ax)
    ax.plot(
        path_list[:, index_1],
        path_list[:, index_2],
        color="black",
        linewidth=1,
        label="Path",
    )
    ax.scatter(
        path_list[0, index_1],
        path_list[0, index_2],
        color="g",
        marker="*",
        s=400,
        label="Initial Params",
    )
    ax.scatter(
        path_list[-1, index_1],
        path_list[-1, index_2],
        color="r",
        marker="*",
        s=400,
        label="Final Params",
    )
    if batch_size is not None:
        ax.scatter(
            path_list[:: 1000 // batch_size, index_1],
            path_list[:: 1000 // batch_size, index_2],
            label=f"Jump across batch size:{batch_size}",
            color="magenta",
            s=30,
        )
    plt.legend()
    ax.set_title(f"Error Contour for the most important parameters in {algo_name}")
    ax.set_xlabel(f"Parameter {index_1}")
    ax.set_ylabel(f"Parameter {index_2}")
    plt.savefig(
        f"./part1_plots/Optimization path with most important params for {algo_name}.png"
    )
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("mid_term_data.csv")
    X = df.drop("eta", axis=1).values
    y = df["eta"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_aug = augment_data(X)
    params_least_squares = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
    print(f"Least Squares Params are {params_least_squares}")
    print(f"Least Squares Error is {error_func(X_aug, y, params_least_squares)}")
    np.savetxt("Least Squares Params.txt", params_least_squares)
    num_epochs = 500

    ## Batch Gradient Descent
    iter_count, final_params, path_list, error_list = batch_gradient_descent(
        X_aug, y, num_epochs=num_epochs
    )
    np.savetxt("Batch Gradient Descent Params.txt", final_params)

    print(f"Batch Gradient Descent took {iter_count} iterations")
    print(f"Final Params for Batch Gradient Descent are {final_params}")
    print(f"Final Error for batch gradient descent is {error_list[-1]}")
    most_important_params = np.argsort(np.abs(final_params))[::-1][:2]
    visualize_path(
        path_list,
        most_important_params[0],
        most_important_params[1],
        "Batch Gradient Descent",
    )
    plt.plot(np.log(error_list))
    plt.xlabel("Iterations")
    plt.ylabel("Log Error")
    plt.title("Error vs Iterations for Batch Gradient Descent")
    plt.savefig("./part1_plots/Error vs Iterations for Batch Gradient Descent.png")
    plt.show()

    batch_size = 20

    ### Mini Batch Gradient Descent
    iter_count, final_params, path_list, error_list = mini_batch_gradient_descent(
        X_aug, y, num_epochs=num_epochs
    )
    np.savetxt("Mini Batch Gradient Descent Params.txt", final_params)
    print(
        f"Mini Batch Gradient Descent took {iter_count} iterations with batch size {batch_size}"
    )
    print(f"Final Params for Mini Batch Gradient Descent are {final_params}")
    print(f"Final Error for Mini Batch Gradient Descent is {error_list[-1]}")
    most_important_params = np.argsort(np.abs(final_params))[::-1][:2]
    visualize_path(
        path_list,
        most_important_params[0],
        most_important_params[1],
        "Mini Batch Gradient Descent",
        batch_size=batch_size,
    )
    for batch_size in [10, 1000]:
        iter_count, final_params, path_list, error_list = mini_batch_gradient_descent(
            X_aug, y, num_epochs=num_epochs, batch_size=batch_size
        )
        plt.plot(np.log(error_list), label=f"Batch Size {batch_size}")
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Log Error")
    plt.title("Error vs Iterations for Mini Batch Gradient Descent")
    plt.savefig("./part1_plots/Error vs Iterations for Mini Batch Gradient Descent.png")
    plt.show()

    batch_list = [10, 1000]
    temp_list = []
    for batch in batch_list:
        iter_count, final_params, path_list, error_list = mini_batch_gradient_descent(
            X_aug, y, num_epochs=num_epochs, batch_size=batch
        )
        temp_list.append(path_list)
    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, path_list in enumerate(temp_list):
        path_list = np.vstack(path_list)
        ax.plot(
            path_list[:, most_important_params[0]],
            path_list[:, most_important_params[1]],
            label=f"Batch Size {batch_list[idx]}",
        )
    plt.legend()
    ax.set_title(
        "Optimization Path for different batch sizes in Mini Batch Gradient Descent"
    )
    ax.set_xlabel(f"Parameter {most_important_params[0]}")
    ax.set_ylabel(f"Parameter {most_important_params[1]}")
    min_x = (
        min(
            np.min(np.vstack(path_list)[:, most_important_params[0]])
            for path_list in temp_list
        )
        - 1
    )
    max_x = (
        max(
            np.max(np.vstack(path_list)[:, most_important_params[0]])
            for path_list in temp_list
        )
        + 1
    )
    min_y = (
        min(
            np.min(np.vstack(path_list)[:, most_important_params[1]])
            for path_list in temp_list
        )
        - 1
    )
    max_y = (
        max(
            np.max(np.vstack(path_list)[:, most_important_params[1]])
            for path_list in temp_list
        )
        + 1
    )

    X_mesh, Y_mesh = np.meshgrid(
        np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100)
    )
    Z = np.zeros_like(X_mesh).flatten()
    for idx in range(X_mesh.flatten().shape[0]):
        params = params_least_squares.copy()
        params[most_important_params[0]] = X_mesh.flatten()[idx]
        params[most_important_params[1]] = Y_mesh.flatten()[idx]
        Z[idx] = error_func(X_aug, y, params)
    Z = Z.reshape(X_mesh.shape)
    contour = ax.contourf(X_mesh, Y_mesh, Z, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.savefig(
        "./part1_plots/Optimization path for different batch sizes in Mini Batch Gradient Descent.png"
    )
    plt.show()

    ### Adam

    iter_count, final_params, path_list, error_list = mini_batch_adam(
        X_aug, y, num_epochs=num_epochs, batch_size=10
    )
    np.savetxt("Mini Batch Adam Params.txt", final_params)
    print(f"Mini Batch Adam took {iter_count} iterations")
    print(f"Final Params for Mini Batch Adam are {final_params}")
    print(f"Final Error for Mini Batch Adam is {error_list[-1]}")

    most_important_params = np.argsort(np.abs(final_params))[::-1][:2]
    visualize_path(
        path_list,
        most_important_params[0],
        most_important_params[1],
        "Mini Batch Adam",
    )
    batch_list = [100, 500, 1000]
    for batch_size in batch_list:
        iter_count, final_params, path_list, error_list = mini_batch_adam(
            X_aug, y, num_epochs=num_epochs, batch_size=batch_size, lr=1e-2
        )
        plt.plot(error_list, label=f"Batch Size {batch_size}")
    plt.xlabel("Iterations")
    plt.legend()
    plt.ylabel("Error")
    plt.title("Error vs Iterations for Mini Batch Adam")
    plt.savefig("./part1_plots/Error vs Iterations for Mini Batch Adam.png")
    plt.show()

    ## Plotting the paths for different batch sizes
    temp_list = []

    for batch in batch_list:
        iter_count, final_params, path_list, error_list = mini_batch_adam(
            X_aug, y, num_epochs=num_epochs, batch_size=batch
        )
        temp_list.append(path_list)
    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, path_list in enumerate(temp_list):
        path_list = np.vstack(path_list)
        ax.plot(
            path_list[:, most_important_params[0]],
            path_list[:, most_important_params[1]],
            label=f"Batch Size {batch_list[idx]}",
        )
    plt.legend()
    ax.set_title("Optimization Path for different batch sizes in Mini Batch Adam")
    ax.set_xlabel(f"Parameter {most_important_params[0]}")
    ax.set_ylabel(f"Parameter {most_important_params[1]}")
    min_x = (
        min(
            np.min(np.vstack(path_list)[:, most_important_params[0]])
            for path_list in temp_list
        )
        - 1
    )
    max_x = (
        max(
            np.max(np.vstack(path_list)[:, most_important_params[0]])
            for path_list in temp_list
        )
        + 1
    )
    min_y = (
        min(
            np.min(np.vstack(path_list)[:, most_important_params[1]])
            for path_list in temp_list
        )
        - 1
    )
    max_y = (
        max(
            np.max(np.vstack(path_list)[:, most_important_params[1]])
            for path_list in temp_list
        )
        + 1
    )
    X_mesh, Y_mesh = np.meshgrid(
        np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100)
    )
    Z = np.zeros_like(X_mesh).flatten()
    for idx in range(X_mesh.flatten().shape[0]):
        params = params_least_squares.copy()
        params[most_important_params[0]] = X_mesh.flatten()[idx]
        params[most_important_params[1]] = Y_mesh.flatten()[idx]
        Z[idx] = error_func(X_aug, y, params)
    Z = Z.reshape(X_mesh.shape)
    contour = ax.contourf(X_mesh, Y_mesh, Z, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.savefig(
        "./part1_plots/Optimization path for different batch sizes in Mini Batch Adam.png"
    )
    plt.show()
