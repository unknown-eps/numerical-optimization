import numpy as np
import matplotlib.pyplot as plt

from datamaker import RegresssionDataMaker

# Create an instance of the class
data_maker = RegresssionDataMaker(100,1,0.0)

# Generate data

X,y,coefs = data_maker.make_data_with_ones()

# Save the data to a csv file
data_maker.save_data(X,y,'data.csv')
data_maker.save_coefs(coefs,'coefs.csv')
#Make a least squares objective function for regression 

def mse_linear_regression(X,y,theta): # Shape of X is (n_samples,n_features)
    n_samples,n_features = X.shape
    return np.sum((X.dot(theta)-y)**2,axis=0)/n_samples

# Make a function to compute the gradient of the least squares objective function

def gradient_mse_linear_regression(X,y,theta): # Shape of X is (n_samples,n_features)
    n_samples,n_features = X.shape
    return 2*X.T.dot(X.dot(theta)-y)/n_samples
# Make a function to perform gradient descent
def finite_differrence(X,y,theta,eps=1e-6):
    n_features = theta.shape[0]
    grad=np.zeros((n_features,1))
    for i in range(n_features):
        theta_plus = theta.copy()
        theta_plus[i]+=eps
        theta_minus = theta.copy()
        theta_minus[i]-=eps
        grad[i] = (mse_linear_regression(X,y,theta_plus)-mse_linear_regression(X,y,theta_minus))/(2*eps)
    return grad
#Auto diff with torch
def automatic_diff(X,y,theta):
    import torch
    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)
    theta_torch = torch.tensor(theta,requires_grad=True)
    y_pred = torch.matmul(X_torch,theta_torch)
    loss = torch.mean((y_pred-y_torch)**2)
    loss.backward()
    return theta_torch.grad.numpy()

step_size = 0.5
n_iterations = 100
theta_0=np.array([[2],[2]])
def gradient_descent(X,y,mse_linear_regression,gradient_mse_linear_regression,step_size,n_iterations,theta_0):
    n_samples,n_features = X.shape
    theta = theta_0
    path=theta
    for i in range(n_iterations):
        if(np.linalg.norm(gradient_mse_linear_regression(X,y,theta))<1e-4):
            break
        theta = theta - step_size*gradient_mse_linear_regression(X,y,theta)
        path = np.hstack((path,theta))
        if(i%10==0):
            print('Iteration:',i,'MSE:',mse_linear_regression(X,y,theta),'theta:',theta.flatten())
    return theta,path

# Plot the contour of the least square objective function

def plot_contour(X,y,mse_linear_regression,path,theta):
    theta0_vals = np.linspace(-5,5,100)
    theta1_vals = np.linspace(-5,5,100)
    J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
    for i,theta0 in enumerate(theta0_vals):
        for j,theta1 in enumerate(theta1_vals):
            J_vals[j,i] = mse_linear_regression(X,y,np.array([[theta0],[theta1]])).item()
    plt.contourf(theta0_vals,theta1_vals,J_vals,levels=np.arange(0,100,10))
    #plot the path
    plt.plot(path[0],path[1],'ro-')
    plt.xlabel('theta_0$')
    plt.ylabel('theta_1')
    plt.title(f'Contour of the least squares objective function with step_legth {step_size}')
    # PLot the iteration number aloingside
    for i in range(0,path.shape[1],10):
        plt.text(path[0,i],path[1,i],str(i))
    #Mark the final point
    plt.plot(theta[0],theta[1],'bo')
    plt.savefig(f"contour with path step_length_{step_size}.png")
    plt.show()

print(theta_0.shape)  
theta,path=gradient_descent(X,y,mse_linear_regression,gradient_mse_linear_regression,step_size,n_iterations,theta_0)
print("Number of iterations:",path.shape[1])
print("Final theta:",theta.flatten())
plot_contour(X,y,mse_linear_regression,path,theta)

def plot_contour_no_path(X,y,mse_linear_regression,save_path):
    theta0_vals = np.linspace(-5,5,100)
    theta1_vals = np.linspace(-5,5,100)
    J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
    for i,theta0 in enumerate(theta0_vals):
        for j,theta1 in enumerate(theta1_vals):
            J_vals[j,i] = mse_linear_regression(X,y,np.array([[theta0],[theta1]])).item()
    plt.contourf(theta0_vals,theta1_vals,J_vals,levels=np.arange(0,100,5),cmap="coolwarm")
    
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.title('Contour of the least squares objective function')
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()
plot_contour_no_path(X[:3,:],y[:3],mse_linear_regression,"contour_no_path_0_to_3.png")
plot_contour_no_path(X[3:6,:],y[3:6],mse_linear_regression,"contour_no_path_3_to_6.png")

# create a contour plotting function that does not save
def plot_contour_subplot(X,y,mse_linear_regression,id):
    theta0_vals=np.linspace(-5,5,100)
    theta1_vals=np.linspace(-5,5,100)
    
    theta_0,theta_1=np.meshgrid(theta0_vals,theta1_vals)
    #Calcukated J_vals using vectorized computation
    theta_0, theta_1 = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = mse_linear_regression(X, y, np.vstack([theta_0.ravel(), theta_1.ravel()])).reshape(theta_0.shape)
    # print(J_vals.shape)
    plt.contourf(theta_0,theta_1,J_vals,levels=np.arange(0,100,10),cmap='coolwarm')
    plt.title(f'Contour_mse_{str(id)}')
# Make a batch selector with batch_size and batch_index as arguments
# The batch_selector mush return X and y from batch_size*(batch_idx-1):batch_index
def batch_selector(X,y,batch_size,batch_index):
    num_samples=X.shape[0]
    start_index=batch_size*batch_index
    end_index=start_index+batch_size
    end_index=min(end_index,num_samples)
    return X[start_index:end_index,:],y[start_index:end_index]

batch_size=5
plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    X_i,y_i=batch_selector(X,y,batch_size,i)
    plot_contour_subplot(X_i,y_i,mse_linear_regression,i)
    plt.tight_layout()
    
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("contour_mse_4*4")
plt.show()


theta_0=np.array([[2],[2]])
step_length=0.001
n_epochs=1000
batch_size=5
num_batches=X.shape[0]//batch_size
alpha=0.0
beta=0.999
beta1=alpha
beta2=beta
def batch_gradient_descent(X,y,mse_linear_regression,gradient_mse_linear_regression,batch_selector,step_length,theta_0,batch_size=5,n_epochs=100,tol=1e-6):
    num_batches=X.shape[0]//batch_size
    dir_descent_store=np.zeros((theta_0.shape[0],num_batches))
    theta=theta_0
    path=theta
    iter_cnt=0
    ave_dir_descent=np.zeros((theta.shape[0],1))
    ave_dir_descent_sq=np.zeros((theta.shape[0],1))
    for epoch in range(n_epochs):
        for batch_index in range(num_batches):
            iter_cnt+=1
            X_batch,y_batch=batch_selector(X,y,batch_size,batch_index)
            dir_descent=gradient_mse_linear_regression(X_batch,y_batch,theta)
            # dir_descent_store[:,batch_index]=dir_descent.flatten()
            # ave_dir_descent=(ave_dir_descent*batch_index+dir_descent)/(batch_index+1)
            ave_dir_descent = beta1 * ave_dir_descent + (1 - beta1) * dir_descent
            ave_dir_descent_corr = ave_dir_descent / (1 - beta1**iter_cnt)
            ave_dir_descent_sq = beta2 * ave_dir_descent_sq + (1 - beta2) * (dir_descent**2)
            ave_dir_descent_sq_corr = ave_dir_descent_sq / (1 - beta2**iter_cnt)
            adaptive_step_length = step_length / (np.sqrt(ave_dir_descent_sq_corr) + 1e-8)
            theta=theta-adaptive_step_length*ave_dir_descent_corr
            path=np.hstack((path,theta))
        if(np.linalg.norm(ave_dir_descent)<tol):    
            break
    return path,iter_cnt
path,iter_cnt=batch_gradient_descent(X,y,mse_linear_regression,gradient_mse_linear_regression,batch_selector,step_length,theta_0,batch_size,n_epochs)
print("Number of iterations:",iter_cnt)
print("Final theta:",path[:,-1])
def print_contour_with_path(X,y,mse_linear_regression,path,id):
    theta0_vals=np.linspace(0,3,100)
    theta1_vals=np.linspace(0,3,100)
    
    theta_0,theta_1=np.meshgrid(theta0_vals,theta1_vals)
    #Calcukated J_vals using vectorized computation
    theta_0, theta_1 = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = mse_linear_regression(X, y, np.vstack([theta_0.ravel(), theta_1.ravel()])).reshape(theta_0.shape)
    # print(J_vals.shape)
    plt.contourf(theta_0,theta_1,J_vals,levels=np.arange(0,100,10),cmap='coolwarm')
    plt.title(f'Contour_mse_{str(id)}')
    #plot the path x and y
    plt.plot(path[0], path[1], 'rx-')
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    X_i,y_i=batch_selector(X,y,batch_size,i)
    print_contour_with_path(X_i,y_i,mse_linear_regression,path,i)
    plt.tight_layout()
    
# plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.savefig(f"contour_mse_4*4_with_path_mod_wavg_alpha_{alpha}_beta_{beta}_sl_{step_length}.png")
plt.show()


def plot_rate_of_convergance(path,theta,id):
    diff=path-np.array([[1],[2]])
    plt.plot(np.linalg.norm(diff,axis=0))
    plt.xlabel('Iteration')
    plt.ylabel('Rate of convergence')
    plt.title(f'Rate of convergence_{str(id)}')
    plt.savefig(f'Rate_of_convergence_{str(id)}.png')
    plt.show()

plot_rate_of_convergance(path,theta,f"alpha_{alpha}_beta_{beta}_sl_{step_length}")