import numpy as np
import matplotlib.pyplot as plt
# Defining the rosenbrock function
def rosenbrock(x :np.ndarray,a=1,b=100) -> float:
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

# Defining the gradient of the rosenbrock function
def grad_rosenbrock(x :np.ndarray,a=1,b=100) -> np.ndarray:
    return np.array([2*(x[0]-a) + 4*b*x[0]*(x[0]**2-x[1]),2*b*(x[1]-x[0]**2)])

# Defining the hessian of the rosenbrock function
def hessian_rosenbrock(x :np.ndarray,a=1,b=100) -> np.ndarray:
    hess=np.zeros((2,2))
    hess[0,0]=2-4*b*x[1] + 12*b*x[0]**2
    hess[0,1]=-4*b*x[0]
    hess[1,0]=-4*b*x[0]
    hess[1,1]=2*b
    return hess

x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Rosenbrock Function')
X,Y=np.meshgrid(x,y)
Z=rosenbrock(np.array([X,Y]))
plt.contourf(X,Y,Z,levels=np.arange(0,100,10),cmap='jet')
plt.colorbar
plt.savefig('rosenbrock.png')
plt.show()

# Apply backtracking gradient descent to the rosenbrock function
def plot_path(path:np.ndarray,a=1,b=100,save_path='rosenbrock_path.png',algorithm='Backtracking Gradient Descent') -> None:
    x=np.linspace(-2,2,100)
    y=np.linspace(-2,2,100)
    X,Y=np.meshgrid(x,y)
    Z=rosenbrock(np.array([X,Y]),a,b)
    plt.contourf(X,Y,Z,levels=np.arange(0,100,10),cmap='jet')
    plt.colorbar
    # plt.plot(path[:,0],path[:,1],'r-o')
    plt.plot(path[:,0],path[:,1],c='w',marker='o')
    plt.plot(path[0,0], path[0,1], 'go', label='Start')
    plt.plot(path[-1,0], path[-1,1], 'o', color='orange', label='Finish')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(f'Path of the optimization algorithm: {algorithm}')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
def backtracking_gradient_descent(x:np.ndarray,alpha=1e-4,beta=0.9,a=1,b=100,max_iter=10000,tol=1e-6) -> np.ndarray:
    path=np.copy(x)
    for i in range(max_iter):
        grad=grad_rosenbrock(x,a,b)
        t=1
        while rosenbrock(x-t*grad,a,b) > rosenbrock(x,a,b) - alpha*t*np.linalg.norm(grad)**2:
            t*=beta
        x=x-t*grad
        path=np.vstack((path,x))
        if np.linalg.norm(grad)<tol:
            break
    return i+1,path
x_start=np.array([0.0,0.0])
iter_cnt,path=backtracking_gradient_descent(np.copy(x_start))
print('Number of iterations for backtracking gradient descent:',iter_cnt)
print('Final solution for backtracking gradient descent:',path[-1])
plot_path(path,save_path='backtracking_gradient_descent_path.png',algorithm='Backtracking Gradient Descent')

# Apply backtracking Newton's method to the rosenbrock function

def backtracking_newton(x:np.ndarray,alpha=1e-4,beta=0.9,a=1,b=100,max_iter=10000,tol=1e-6) -> np.ndarray:
    path=np.copy(x)
    for i in range(max_iter):
        grad=grad_rosenbrock(x,a,b)
        hess=hessian_rosenbrock(x,a,b)
        t=1
        while rosenbrock(x-(t*np.linalg.inv(hess)@grad).reshape(-1),a,b) > rosenbrock(x,a,b) - alpha*t*np.linalg.norm(grad)**2:
            t*=beta
        x=x-t*np.linalg.solve(hess,grad)
        path=np.vstack((path,x))
        if np.linalg.norm(grad)<tol:
            break
    return i+1,path
iter_cnt,path=backtracking_newton(np.copy(x_start))
print('Number of iterations for backtracking Newton:',iter_cnt)
print("Final soln of backtracking Newton's method:",path[-1])
plot_path(path,save_path='backtracking_newton_path.png',algorithm="Backtracking Newton's Method")

# Apply symmetric SR-1 method to the rosenbrock function
def symmetric_SR1_with_backtracking(x:np.ndarray,alpha=1e-4,beta=0.9,a=1,b=100,max_iter=1000,tol=1e-6) -> np.ndarray:
    path=np.copy(x)
    H_inv=np.eye(2)
    grad=grad_rosenbrock(x,a,b)
    for i in range(max_iter):
        descent_dir=-H_inv@grad
        descent_dir=descent_dir.reshape(-1)
        if(np.dot(grad,descent_dir)>0):
            # print('Not a descent direction') # If we end in a non-descent direction, we reinitialize the Hessian approximation
            # return i+1,path
            H_inv=np.eye(2)
            descent_dir=-H_inv@grad
        t=1
        while rosenbrock(x+t*descent_dir,a,b) > rosenbrock(x,a,b) - alpha*t*np.dot(grad,descent_dir):
            t*=beta
        x_new=x+t*descent_dir
        grad_new=grad_rosenbrock(x_new,a,b)
        s=x_new-x
        y=grad_new-grad
        rho=np.dot(y,s-(H_inv@y).reshape(-1))
        if(np.abs(rho)>1e-8):
            H_inv=H_inv + np.outer(s-H_inv@y,s-H_inv@y)/rho
        x=x_new
        grad=grad_new
        path=np.vstack((path,x))
        if np.linalg.norm(grad)<tol:
            break
    return i+1,path
iter_cnt,path=symmetric_SR1_with_backtracking(np.copy(x_start))
print('Number of iterations for symmetric SR1:',iter_cnt)
print("Final soln of symmetric SR1 method:",path[-1])
plot_path(path,save_path='symmetric_SR1_with_backtrack_path.png',algorithm='Symmetric SR1 Method')

# Apply BFGS method to the rosenbrock function with backtracking
def bfgs_with_backtracking_line_search(x:np.ndarray,alpha=1e-4,beta=0.9,a=1,b=100,max_iter=1000,tol=1e-6) -> np.ndarray:
    path=np.copy(x)
    H_inv=np.eye(2)# Initialize the inverse Hessian approximation to be updated via BFGS formula
    grad=grad_rosenbrock(x,a,b)
    for i in range(max_iter):
        descent_dir=-H_inv@grad
        descent_dir=descent_dir.reshape(-1)
        t=1
        while rosenbrock(x+t*descent_dir,a,b) > rosenbrock(x,a,b) - alpha*t*np.dot(grad,descent_dir):
            t*=beta
        x_new=x+t*descent_dir
        grad_new=grad_rosenbrock(x_new,a,b)
        s=x_new-x
        y=grad_new-grad
        pho=1/np.dot(y,s)
        H_inv=(np.eye(2)-pho*np.outer(s,y))@H_inv@(np.eye(2)-pho*np.outer(y,s)) + pho*np.outer(s,s)
        x=x_new
        grad=grad_new
        path=np.vstack((path,x))
        if np.linalg.norm(grad)<tol:
            break
    return i+1,path
iter_cnt,path=bfgs_with_backtracking_line_search(np.copy(x_start))
print('Number of iterations for BFGS with backtracking:',iter_cnt)
print("Final soln of BFGS method with backtracking:",path[-1])
plot_path(path,save_path='bfgs_with_backtrack_path.png',algorithm='BFGS Method with Backtracking')