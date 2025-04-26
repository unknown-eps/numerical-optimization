import numpy as np
from scipy.optimize import linprog
## Constants
INTEREST_RATE = 0.005  # Interest rate for investment
K_REPAY_FACTOR = 1.025  # Repayment factor for 4-month loan
J_REPAY_FACTOR = (1+0.018)**2  # Repayment factor for 6-month loan
L_REPAY_FACTOR = (1+0.01)**8  # Repayment factor for 2-year loan
cash_flow = [100, 500, 100, -600, -500, 200, 600, -900]

# Decision Variables
# L: Two-year loan available at the beginning of the first quarter # Index 0
# x_t: Total money at the start of quarter t after all loans have also been taken. t = 1 to 9 # Index 1 to 9
# i_t: The amount invested in quarter t. t = 1 to 8 # Index 1 to 8 # Index 10 to 17
# j_t: 6-month loan available at the beginning of each quarter t, t = 1 to 7 # Index 1 to 7 # Index 18 to 24
# k_t: 4-month loan available at the beginning of each quarter t, t = 1 to 8 # Index 1 to 8 # Index 25 to 32

# Total decision variables = 1 (L) +  9 (x_t)+ 8 (i_t) + 7 (j_t) + 8 (k_t)  = 33
# We already have non-negativity constraints for all decision variables.
# The objective function is  x_9 - L_REPAY_FACTOR * L
# The number of constraints are 17 without counting the non-negativity constraints.


def simplex_iteration(A, b, c, basic_vars, tol=1e-8,xb=None):
    '''
    Perform one iteration of the simplex algorithm.
    A: Coefficient matrix
    b: Right-hand side vector
    c: Cost vector
    basic_vars: Indices of basic variables
    tol: Tolerance for numerical stability
    Returns:
    Returns the function value for the choice of the basic variables,new values of the basic variables ,new basic variables and True if the solution is optimal, False otherwise.

    '''
    m, n = A.shape
    non_basic_vars = [temp for temp in range(n) if temp not in basic_vars]
    B = A[:, basic_vars]
    NB = A[:, non_basic_vars]
    if xb is None:
        xb = np.linalg.solve(B,b) # Since this is the first time we explicitly solve the system of equations
    lamb = np.linalg.solve(B.T, c[basic_vars])
    s_nb = c[non_basic_vars] - NB.T @ lamb
    if np.all(s_nb >= -tol):
        return np.dot(c[basic_vars], xb),xb,basic_vars, True
    
    # According to Bland's rule, select the smallest index variable with negative reduced cost
    entering_idx = None
    for i, j in enumerate(non_basic_vars):
        if s_nb[i] < -tol:
            entering_idx = i
            break  # Take the first negative reduced cost variable (smallest index)
    
    # Calculate direction
    d = np.linalg.solve(B, NB[:, entering_idx])
    
    # Check if unbounded
    if np.all(d <= tol):
        raise Exception("Problem is unbounded")
    
    # Find the leaving variable according to Bland's rule (minimum ratio test)
    min_ratio = float('inf')
    leaving_idx = None
    for i in range(m):
        if d[i] > tol:
            ratio = xb[i] / d[i]
            if ratio < min_ratio or (abs(ratio - min_ratio) < tol and basic_vars[i] < basic_vars[leaving_idx]):
                min_ratio = ratio
                leaving_idx = i
    
    # Update the basic variables
    # Update the basic variables
    new_xb = xb.copy()
    for i in range(m):
        if i == leaving_idx:
            new_xb[i] = min_ratio
        else:
            new_xb[i] = xb[i] - d[i] * min_ratio
    entering_var = non_basic_vars[entering_idx]
    new_basic_vars = basic_vars.copy()
    new_basic_vars[leaving_idx] = entering_var
    
    # sort the basic variables
    combined = list(zip(new_basic_vars, new_xb))
    combined.sort(key=lambda x: x[0])  # Sort by basic variable index
    new_basic_vars_sorted = [var for var, _ in combined]
    new_xb_sorted = [val for _, val in combined]
    return np.dot(c[basic_vars], xb), new_xb_sorted, new_basic_vars_sorted, False





def direct_solve(A, b, c):
    A = A.copy()
    b = b.copy()
    c = c.copy()
    c_min = c
    
    # Define bounds for z >= 0
    num_vars = A.shape[1]
    bounds = [(0, None)] * num_vars
    
    # Solve the linear programming problem
    res = linprog(c=c_min, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    
    # Check if the solver was successful
    if not res.success:
        raise ValueError(f"Linear programming solver failed: {res.message}")
        
    # The optimal objective value for the maximization problem is -res.fun
    # The optimal solution vector is res.x
    return res.x, -res.fun, res
    
def find_basic_vars_explicit(A, b):
    m, n = A.shape  # 17 constraints, 33 original vars
    A_artificial = np.eye(m)
    A_aug = np.hstack([A, A_artificial])  # shape (17, 50)
    
    c_phase1 = np.zeros(n + m)
    c_phase1[n:] = 1  # Cost vector: minimize sum of artificial variables

    bounds = [(0, None)] * (n + m)  # Non-negative variables

    # Solve Phase 1
    res = linprog(c=c_phase1, A_eq=A_aug, b_eq=b, bounds=bounds, method="highs")

    if not res.success:
        raise ValueError("Phase 1 problem is infeasible, cannot find initial basic variables.")

    x = res.x  # Solution vector of length 50

    # Now find the basic variables among original vars (first 33)
    basic_vars = []
    for i in range(n):
        if np.isclose(x[i], 0):
            continue
        basic_vars.append(i)
    
    return basic_vars, res

L_start_idx = 0
X_START_IDX = 1
I_START_IDX = 10
J_START_IDX = 18
K_START_IDX = 25
A = np.zeros((17, 33))
b = np.zeros(17)
for t in range(0,9):
    cur_L_idx = L_start_idx
    cur_x_idx = t + X_START_IDX
    cur_i_idx = t + I_START_IDX
    cur_j_idx = t + J_START_IDX
    cur_k_idx = t + K_START_IDX
    A[t, cur_x_idx] = -1
    if t==0:
        A[t, cur_L_idx] = 1
    if t not in [7,8]: # # No 6-month loan taken at start of Q8 and while calculating x[9]
        A[t, cur_j_idx] = 1
    if t!=8: ## No 4-month loan caculation at start of Q9
        A[t, cur_k_idx] = 1
    if t>0:
        A[t, cur_x_idx-1] = 1
        A[t, cur_i_idx-1] = INTEREST_RATE
        A[t, cur_k_idx-1] = -K_REPAY_FACTOR
        if t>1:
            A[t, cur_j_idx-2] = -J_REPAY_FACTOR
        b[t] = cash_flow[t-1]
        
for idx in range(9, 17):
    t = idx - 9
    cur_i_idx = t + I_START_IDX
    cur_x_idx = t + X_START_IDX
    A[idx, cur_x_idx] = 1
    A[idx, cur_i_idx] = -1
    b[idx] = cash_flow[t]
print(f"Rank of the matrix A: {np.linalg.matrix_rank(A)}") # Rank of the matrix A: 17 Full row rank
c = np.zeros(33)
c[L_start_idx] = +L_REPAY_FACTOR
c[X_START_IDX + 8] = -1

## Direct solve via scipy
solution_by_direct_solve, obj_val, res = direct_solve(A, b, c)
print(f"Solution by direct solve has the optimal value: {obj_val:.2f}")

## Getting an initial basic feasible solution
basic_vars, res = find_basic_vars_explicit(A, b)
print(f"Basic variables: {basic_vars}")
basic_vars.append(31) # This was chosen through hit and trial so that the matrix is full rank 

max_iter = 10

for i in range(max_iter):
    print(f"Iteration {i+1}")
    obj_val, xb, basic_vars, is_optimal = simplex_iteration(A, b, c, basic_vars)
    print(f"Objective value: {obj_val:.2f}")
    print(f"Basic variables: {basic_vars}")
    print(f"xb: {xb}")
    print("\n***Next iteration*** \n")
    if is_optimal:
        break