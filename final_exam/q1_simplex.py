import numpy as np
<<<<<<< HEAD
from scipy.optimize import linprog
=======
>>>>>>> q1
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
<<<<<<< HEAD



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

=======
>>>>>>> q1
L_start_idx = 0
X_START_IDX = 1
I_START_IDX = 10
J_START_IDX = 18
K_START_IDX = 25
A = np.zeros((17, 33))
b = np.zeros(17)
<<<<<<< HEAD
for t in range(0,9):
=======
for t in range(0,10):
>>>>>>> q1
    cur_L_idx = L_start_idx
    cur_x_idx = t + X_START_IDX
    cur_i_idx = t + I_START_IDX
    cur_j_idx = t + J_START_IDX
<<<<<<< HEAD
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
    if cash_flow[t] >=0:
        A[idx, cur_x_idx] = 1
        A[idx, cur_i_idx] = -1
        b[idx] = cash_flow[t]
    else:
        A[idx, cur_x_idx] = 1
        A[idx, cur_i_idx] = -1
        b[idx] = 0
print(f"Rank of the matrix A: {np.linalg.matrix_rank(A)}") # Rank of the matrix A: 17 Full row rank
c = np.zeros(33)
c[L_start_idx] = +L_REPAY_FACTOR
c[X_START_IDX + 8] = -1

solution_by_direct_solve, obj_val, res = direct_solve(A, b, c)
print(f"Solution by direct solve has the optimal value: {obj_val:.2f}")

basic_vars, res = find_basic_vars_explicit(A, b)
print(f"Basic variables: {basic_vars}")
=======
    cur_k_idx = t + x_start_idx + 25
    if t==0:
        A[t, cur_x_idx] = -1
        A[t, cur_L_idx] =  1
        A[t, cur_j_idx] =  1
        A[t, cur_k_idx] =  1
        b[t] = 0
    elif t==1:
        A[t, cur_x_idx] = -1
        A[t, cur_i_idx-1] = INTEREST_RATE
        A[t, cur_k_idx-1] = -K_REPAY_FACTOR
        A[t, cur_k_idx] = 1
        A[t, cur_j_idx] = 1
        b[t] = cash_flow[0]
    elif t in range(2,8):
        A[t, cur_x_idx] = -1
        A[t, cur_i_idx-1] = INTEREST_RATE
        A[t, cur_k_idx-1] = -K_REPAY_FACTOR
        A[t, cur_j_idx-2] = -J_REPAY_FACTOR
        A[t, cur_j_idx] = 1
        A[t, cur_k_idx] = 1
        b[t] = cash_flow[t-1]
    elif 
>>>>>>> q1
