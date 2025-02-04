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
    return np.sum((X.dot(theta)-y)**2)/n_samples

# Make a function to compute the gradient of the least squares objective function

def gradient_mse_linear_regression(X,y,theta): # Shape of X is (n_samples,n_features)
    n_samples,n_features = X.shape
    return 2*X.T.dot(X.dot(theta)-y)/n_samples
# Make a function to perform gradient descent

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
    plt.contourf(theta0_vals,theta1_vals,J_vals,levels=np.arange(0,100,10),cmap="coolwarm")
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.title('Contour of the least squares objective function')
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()
plot_contour_no_path(X[:3,:],y[:3],mse_linear_regression,"contour_no_path_0_to_3.png")
plot_contour_no_path(X[3:6,:],y[3:6],mse_linear_regression,"contour_no_path_3_to_6.png")