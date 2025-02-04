# Code a backtracking line search algorithm
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D callable function representing x[0]**2 + 2*x[1]**2

def f(x):
    return x[0]**2+x[1]**2+0.5*x[0]*x[1]
def grad_f(x):
    return np.array([2*x[0]+0.5*x[1],2*x[1]+0.5*x[0]])

# 1D line search algorithm
def backtracking_line_search(f,grad_f,direction_descent,x, alpha=1 , pho=0.8,c=1e-4):
    step_length=alpha
    while f(x+step_length*direction_descent)>f(x)+c*step_length*np.dot(grad_f(x),direction_descent):
        step_length=pho*step_length
    return step_length

# Define is a given direction of descent is a valid descent direction
def is_descent_direction(f,grad_f,direction_descent,x):
    return np.dot(grad_f(x),direction_descent)<0

# Define a function to perform gradient descent
def gradient_descent(f,grad_f,x_0,alpha,rho,max_iter=300,tol=1e-6):
    x = x_0
    path=x
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        direction_descent = -grad_f(x)
        step_length=backtracking_line_search(f,grad_f,direction_descent,x,alpha,rho)
        x = x+step_length*direction_descent
        path=np.vstack((path,x))
    return x, f(x), ii, path

x=np.linspace(-10,10,100)
y=np.linspace(-10,10,100)
X,Y=np.meshgrid(x,y)
Z=f([X,Y])

cfp=plt.contourf(X,Y,Z,levels=np.linspace(0,100,10),cmap='Blues',extend='max',vmin=0,vmax=100)
plt.colorbar(cfp)
plt.show()
plt.savefig("f_contour.png")

# Set initial guess as x_0=[5,5]
x_0=np.array([5,5])

# Perform gradient descent
alpha=1
pho=0.5
x, f_x, num_iter, path = gradient_descent(f,grad_f,x_0,alpha,pho)
print("Number of iterations: ", num_iter)
# VISUALIZE THE PATH

plt.contourf(X,Y,Z,levels=np.linspace(0,100,10),cmap='Blues',extend='max',vmin=0,vmax=100)
plt.plot(path[:,0],path[:,1])
#Mark the initial and final points in red and green respectively

plt.plot(path[0,0],path[0,1],'ro')
plt.plot(path[-1,0],path[-1,1],'go')
# Set the title of the plot and the x,y directions
plt.title(f"Contour of f with path of gradient descent with backtracking alpha_{alpha}_pho_{pho}")
plt.xlabel("x1")
plt.ylabel("x2")
for i, point in enumerate(path):
    plt.text(point[0], point[1], str(i), fontsize=8, color='black')
plt.savefig(f"f_contour_path_alpha_{alpha}_pho_{pho}.png")
plt.show()
# determine step_length and pass it as a hyperparamter to the gradient descent function
def gradient_descent_nobacktracking(f,grad_f,x_0,step_length=0.1,max_iter=300,tol=1e-6):
    x = x_0
    path=x
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        direction_descent = -grad_f(x)
        x = x+step_length*direction_descent
        path=np.vstack((path,x))
    return x, f(x), ii, path

x_0=np.array([5,5])
step_length=0.1
#perfrom gradient descent
x, f_x, num_iter, path = gradient_descent_nobacktracking(f,grad_f,x_0,step_length)
print("Number of iterations without backtracking: ", num_iter)
print("Final solution without backtracking: ", x)
# VISUALIZE THE contour
plt.contourf(X,Y,Z,levels=np.linspace(0,100,10),cmap='Blues',extend='max',vmin=0,vmax=100)
#VIsualize the path
plt.plot(path[:,0],path[:,1])
#Mark the initial and final points in red and green respectively
plt.plot(path[0,0],path[0,1],'ro')
plt.plot(path[-1,0],path[-1,1],'go')
#Set the title of the plot and the x,y directions
plt.title("Contour of f with path of gradient descent without backtracking")
plt.xlabel("x1")
plt.ylabel("x2")
#Mark all the ponts in the path with their respective index
for i, point in enumerate(path):
    plt.text(point[0], point[1], str(i), fontsize=8, color='black')
plt.savefig("f_contour_path_without_backtracking")
plt.show()

### HW:Random descent direction
# Main function modifiing gradient descent
np.random.seed(1) # Set seed for reproducibility
def random_gradient_descent(f,grad_f,x_0,alpha,rho,max_iter=300,tol=1e-6):
    x = x_0
    path=x
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        direction_descent = np.random.randn(2)
        while not is_descent_direction(f,grad_f,direction_descent,x):
            direction_descent = np.random.randn(2)
        step_length=backtracking_line_search(f,grad_f,direction_descent,x,alpha,rho)
        x = x+step_length*direction_descent
        path=np.vstack((path,x))
    return x, f(x), ii, path
x_0=np.array([5,5])
alpha=1
pho=0.8
x_optim, f_x, num_iter, path = random_gradient_descent(f,grad_f,x_0,alpha,pho)
print("Number of iterations for randomized gradient descent: ", num_iter)
print("Final solution for randomized gradient descent: ", x_optim)
plt.plot(path[:,0],path[:,1])
plt.contourf(X,Y,Z,levels=np.linspace(0,100,10),cmap='Blues',extend='max',vmin=0,vmax=100)
plt.plot(path[0,0],path[0,1],'ro')
plt.plot(path[-1,0],path[-1,1],'go')
plt.title("Contour of f with path of randomized gradient descent")
plt.xlabel("x1")
plt.ylabel("x2")
for i, point in enumerate(path):
    plt.text(point[0], point[1], str(i), fontsize=8, color='black')
plt.savefig("f_contour_path for randomized_gradient_descent")
plt.show()

#HW Co-oridnate descent
x_0=np.array([5,5])
def coordinate_descent(f,grad_f,x_0,alpha,rho,max_iter=300,tol=1e-6):
    x = x_0
    path=x
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        for jj in range(len(x)):
            direction_descent = np.zeros(len(x))
            direction_descent[jj] = 1
            if not is_descent_direction(f,grad_f,direction_descent,x):
                direction_descent[jj] = -1
                if not is_descent_direction(f,grad_f,direction_descent,x):
                    continue
            step_length=backtracking_line_search(f,grad_f,direction_descent,x,alpha,rho)
            x = x+step_length*direction_descent
            path=np.vstack((path,x))
    return x, f(x), ii, path
alpha=1
pho=0.8
x_optim, f_x, num_iter, path = coordinate_descent(f,grad_f,x_0,alpha,pho)
print("Number of iterations for coordinate descent: ", num_iter)
print("Final solution for coordinate descent: ", x_optim)
plt.plot(path[:,0],path[:,1])
plt.contourf(X,Y,Z,levels=np.linspace(0,100,10),cmap='Blues',extend='max',vmin=0,vmax=100)
plt.plot(path[0,0],path[0,1],'ro')
plt.plot(path[-1,0],path[-1,1],'go')
plt.title("Contour of f with path of coordinate descent")
plt.xlabel("x1")
plt.ylabel("x2")
for i, point in enumerate(path):
    plt.text(point[0], point[1], str(i), fontsize=8, color='black')
plt.savefig("f_contour_path for coordinate_descent")
plt.show()