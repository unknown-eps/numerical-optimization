import numpy as np
import matplotlib.pyplot as plt

# Create a 1D scalar function 

def f(x):
    return 2*x**2+3*x+4

# Visualize this functionb for x in [-10, 10]


x=np.linspace(-10,10,100)
y=f(x)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("f_x.png")
plt.show()
# define derivative of f(x)

def grad_f(x):
    return 4*x+3

# make an initial guess of x_optim as x=5

x_0=5

print(f"x_0={x_0}")
print(f"f_grad at {x_0} = {grad_f(x_0)}")

#dir_descent 

# dir_descent=-grad_f(x_0)

# Let us consider a step length that determines the amount of movement in the direction of the descent

# step_length

# x_new=x_0+step_length*dir_descent

# What is the function value f(x_new) at this new point x_new?

# total_movement =  stelength*dir_descent

# Take a first order approximationj of f(x_new) around x_0

# f_x_new=f(x_0)+grad_f(x_0)*step_length*dir_descent + O(total_movement^2)

# f_x_new(step_length)=f(x_0) + grad_f(x_0) * step_length * dir_descent

# Goal: to find step_length such that f_x__new(step_length) is minimized

# step_length_optim = argmin(f_x_new(step_length))

# step_length shoucld be decided such that there is sufficient decrease in f(x) from x_0 to x_new

# Consider a f(x1,x2) = x1^2 + x2^2

def f2(x1,x2):
    return x1**2+x2**2+0.5*x1*x2

# Visualize this function for x1 in [-10,10] and x2 in [-10,10]

x1=np.linspace(-4,4,100)
x2=np.linspace(-4,4,100)
[X1,X2]=np.meshgrid(x1,x2)
Y=f2(X1,X2)
cfp = plt.contourf(X1, X2, Y, levels=np.linspace(0, 10, 10), cmap='Blues', extend='max', vmin=0, vmax=10)
plt.colorbar(cfp)
plt.xlabel("x1")
plt.ylabel("x2")
#Use matplotlib olormap blues
# plt.clim(0,10)
# plt.set_cmap("Blues")
#Set color axis limits to 0,10
plt.savefig("f2_x1_x2_contour.png")
plt.show()
def grad_f2(x1,x2):
    return np.array([2*x1+0.5*x2,2*x2+0.5*x1])
#HW Solve for the stationary points of f2(x1,x2) by line search
# Define the gradient of f2(x1,x2)
# Solve for stationary points of f2(x1,x2) by line search
max_iter=100
path=[[3,3]]
for _ in range (max_iter):
    descent_dir=-grad_f2(path[-1][0],path[-1][1])
    if(np.linalg.norm(descent_dir)<1e-3):
        break
    cur_point=np.array((path[-1][0],path[-1][1]))
    step_length=1
    while f2(cur_point[0] + step_length * descent_dir[0], cur_point[1] + step_length * descent_dir[1]) >= f2(cur_point[0], cur_point[1]):
        step_length *= 0.9
    new_point = cur_point + step_length * descent_dir
    path.append(new_point.tolist())
print(f"Number of iterations: {len(path)}")
plt.contourf(X1, X2, Y, levels=np.linspace(0, 10, 10), cmap='Blues', extend='max', vmin=0, vmax=10)
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'ro-', label='Gradient Descent Path')
plt.legend()
plt.plot(path[0, 0], path[0, 1], 'go', label='Start')  # Mark the start point
plt.plot(path[-1, 0], path[-1, 1], 'bo', label='End')  # Mark the end point
plt.savefig("gradient_descent_path.png")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()