#Lec_Linear_Program

# The problem statement
# Obj function min x[0] + 2*x[1]
# Subject to x[0]<=1
# x[0]+x[1]<=2
# x[0],x[1]>=0

import numpy as np
A = np.array([[1,0],[1,1]]) # Coefficients of the constraints
b = np.array([1,2]).reshape(-1,1) # RHS of the constraints
c = np.array([1,2]).reshape(-1,1) # Coefficients of the objective function

A = np.hstack((A,np.eye(2))) # Add slack variables
c = np.vstack((c,np.zeros((2,1)))) # Add zeros for the slack variables

m,n = A.shape

# Make a list of index of basic variables and non-basic variables

basic_vars = np.arange(0,n-m) # A list of indices of basic variables shape = (m,)
non_basic_vars = np.arange(n-m,n) # A list of indices of non-basic variables shape = (n-m,)

iter = 0

while True:
    B = A[:,basic_vars] # Basic matrix shape = (m,m)
    N = A[:,non_basic_vars] #Non basic matrix shape = (m,n-m)
    x_B = np.linalg.solve(B,b) # Basic variables values shape = (m,1)

    c_B = c[basic_vars]  # Coefficients of the basic variables shape = (m,1)
    c_N = c[non_basic_vars] # Coefficients of the non-basic variables shape = (n-m,1)
    if np.any(x_B<0):
        print("Infeasible")
        exit()

    lambda_ = np.linalg.inv(B.T)@c_B # Lagrange multipliers shape = (m,1) corresponding to the basic variables

    s_N = c_N - N.T@lambda_ # Reduced costs of the non-basic variables shape = (n-m,1)

    print("Iteration ",iter)

    print("s_N = ",s_N)
    
    
    if np.all(s_N>=0):
        print("Optimal solution found")
        print("x_B = ",x_B)
        print('basic_vars:',basic_vars)
        exit()
    entering_var_index = np.argmin(s_N)
    entering_var = non_basic_vars[entering_var_index]

    print("Entering variable is ",entering_var)

    Aq = A[:,entering_var].reshape(-1,1) # Coefficients of the entering variable shape = (m,1)
    d = np.linalg.solve(B,Aq) # Direction vector shape = (m,1)

    # Find the leaving variable
    mask = d>0
    ratios = x_B[mask]/d[mask]
    
    print("Ratios = ",ratios)
    # ratios = np.array([x_B[i]/d[i] for i in range(m) if d[i]>0 else np.inf])

    basic_vars_mask = basic_vars[mask.ravel()]

    print("Basic vars mask = ",basic_vars_mask)

    mask_leaving_var_index = np.argmin(ratios)
    leaving_var = basic_vars_mask[mask_leaving_var_index]
    print("Leaving variable is ",leaving_var)


    basic_vars = np.delete(basic_vars,leaving_var)
    basic_vars = np.append(basic_vars,entering_var)
    non_basic_vars[entering_var_index] = leaving_var

    basic_vars.sort()
    non_basic_vars.sort()
    
    print("Basic variables after iter ",basic_vars)

    print("Non-basic variables after iter ",non_basic_vars)
    
    iter += 1