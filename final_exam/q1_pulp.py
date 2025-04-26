# My approach
# Assumptions
# - We pay compound interest with the principal at the end of the loan term
# - We cannot take a 6 month loan at the beginning of the 8th quarter as this will be repayed after the ninth quarter
# - If the cashflow is negative(we earn money) for a quarter it cannot be invested in the same quarter

## Variables
# x_t: Total money at the start of quarter t after all loans have also been taken.  
# L: Two-year loan available at the beginning of the first quarter
# i_t: The amount invested in quarter t.
# j_t: 6-month loan available at the beginning of each quarter t  
# k_t: 4-month loan available at the beginning of each quarter t    
# c_t: Cash flow in quarter t provided in the table (Technically c_t is not a decision variable as its values are provided) 
# r: Final result (e.g., total return) to be maximized
## Constants
INTEREST_RATE = 0.005  # Interest rate for investment
K_REPAY_FACTOR = 1.025  # Repayment factor for 4-month loan
J_REPAY_FACTOR = (1+0.018)**2  # Repayment factor for 6-month loan
# Constraints

# Cash flow values (c_t) - Negative means outflow, Positive means inflow

# General variable constraints
# x_t >= 0 for t = 1 to 9 (Total money at start of quarter t after loans)
# i_t >= 0 for t = 1 to 8 (Investment in quarter t)
# j_t >= 0 for t = 1 to 7 (6-month loan taken at start of quarter t)
# k_t >= 0 for t = 1 to 8 (4-month loan taken at start of quarter t)
# L >= 0   (Two-year loan taken at start of quarter 1)

# Investment and Cash Flow constraints
# x_t >= i_t for t = 1 to 8 (Cannot invest more than available cash)
# x_t - i_t >= c_t for t = 1 to 8 (Cash after investment must cover net cash flow)
# Note: If c_t is positive (inflow), this means x_t - i_t >= c_t.
# If c_t is negative (outflow), e.g. c_1 = -100, this means x_t - i_t >= -100,
# or x_t - i_t + 100 >= 0, meaning cash after investment plus the absolute outflow must be non-negative.

# Quarter-specific constraints (Balance Equations)
# Let x[t], i[t], j[t], k[t] represent the variables for quarter t.
# c[t] represents the cash flow for quarter t.

# t=1: Start of Quarter 1
# x[1] = L + j[1] + k[1]

# t=2: Start of Quarter 2
# x[2] = x[1] - c[1] + I_INTEREST_RATE * i[1] - K_REPAY_FACTOR * k[1] + j[2] + k[2]

# t=3: Start of Quarter 3
# x[3] = x[2] - c[2] + I_INTEREST_RATE * i[2] - K_REPAY_FACTOR * k[2] - J_REPAY_FACTOR * j[1] + j[3] + k[3]

# t=4: Start of Quarter 4
# x[4] = x[3] - c[3] + I_INTEREST_RATE * i[3] - K_REPAY_FACTOR * k[3] - J_REPAY_FACTOR * j[2] + j[4] + k[4]

# t=5: Start of Quarter 5
# x[5] = x[4] - c[4] + I_INTEREST_RATE * i[4] - K_REPAY_FACTOR * k[4] - J_REPAY_FACTOR * j[3] + j[5] + k[5]

# t=6: Start of Quarter 6
# x[6] = x[5] - c[5] + I_INTEREST_RATE * i[5] - K_REPAY_FACTOR * k[5] - J_REPAY_FACTOR * j[4] + j[6] + k[6]

# t=7: Start of Quarter 7
# x[7] = x[6] - c[6] + I_INTEREST_RATE * i[6] - K_REPAY_FACTOR * k[6] - J_REPAY_FACTOR * j[5] + j[7] + k[7]

# t=8: Start of Quarter 8
# j[8] = 0 (Constraint: No 6-month loan taken at start of Q8)
# x[8] = x[7] - c[7] + I_INTEREST_RATE * i[7] - K_REPAY_FACTOR * k[7] - J_REPAY_FACTOR * j[6] + k[8] # j[8] is zero

# t=9: Start of Quarter 9 (End state)
# No new loans (j[9], k[9] are implicitly zero as they are not defined/needed)
# x[9] = x[8] - c[8] + I_INTEREST_RATE * i[8] - K_REPAY_FACTOR * k[8] - J_REPAY_FACTOR * j[7]

# Objective Function
# Maximize r = x[9] - L_REPAY_FACTOR * L
# (Maximize the cash at the start of quarter 9 after repaying the initial two-year loan L with interest)

import pulp

# Create the LP problem
lpp = pulp.LpProblem("Cash_Flow_Management", pulp.LpMaximize)

# Define the given cash flow requirements
cash_flow = [None,100, 500, 100, -600, -500, 200, 600, -900]
quarters = range(1, 9)  # Quarters 1 through 8

# Define variables
L = pulp.LpVariable("L", lowBound=0)  # 2-year loan available at beginning of Q1
x = {t: pulp.LpVariable(f"x_{t}", lowBound=0) for t in range(1, 10)}  # Total money at start of quarter t
i = {t: pulp.LpVariable(f"i_{t}", lowBound=0) for t in quarters}  # Amount invested in quarter t
j = {t: pulp.LpVariable(f"j_{t}", lowBound=0) for t in quarters}  # 6-month loan at beginning of quarter t
k = {t: pulp.LpVariable(f"k_{t}", lowBound=0) for t in quarters}  # Quarterly loan at beginning of quarter t
r = pulp.LpVariable("r")  # Final return to be maximized

# Objective function: Maximize wealth at beginning of Q9
lpp += r, "Maximize_Wealth"

# Constraint for r (final return)
lpp += r == x[9] - (1 + 0.01) ** 8 * L, "Final_Return_Constraint"

# Quarter 1 constraints
lpp += x[1] == L + j[1] + k[1], "Q1_Money_Balance"
lpp += x[2] == x[1] - cash_flow[1] + 0.005 * i[1] - 1.025 * k[1] + j[2] + k[2], "Q1_to_Q2_Balance"

# Constraints for quarters 2 to 7
for t in range(2, 8):
    lpp += x[t+1] == x[t] - cash_flow[t] + 0.005 * i[t] - 1.025 * k[t] - (1 + 0.018)**2 * j[t-1] + j[t+1] + k[t+1], f"Q{t}_to_Q{t+1}_Balance"

# Constraint for quarter 8
lpp += x[9] == x[8] - cash_flow[8] + 0.005 * i[8] - 1.025 * k[8] - (1 + 0.018)**2 * j[7], "Q8_to_Q9_Balance"

# No 6-month loan in final quarter
lpp += j[8] == 0, "No_6month_Loan_Q8"

# Investment constraints: can't invest more than what we have after meeting cash requirements
for t in quarters:
    if cash_flow[t] >=0:
        lpp += x[t] - i[t] == cash_flow[t], f"Cash_Flow_Requirement_Q{t}"
    else:
        lpp += i[t] == x[t], f"Investment_Limit_Q{t}"

# Solve the problem
lpp.solve()

# Print the status of the solution
print("Status:", pulp.LpStatus[lpp.status])

# Print the optimal values of the variables
print("\nOptimal Solution:")
print(f"2-Year Loan (L): {pulp.value(L):.2f}")
for t in quarters:
    print(f"\nQuarter {t}:")
    print(f"Total Money (x_{t}): {pulp.value(x[t]):.2f}")
    print(f"Cash Flow (c_{t}): {cash_flow[t]:.2f}")
    print(f"Investment (i_{t}): {pulp.value(i[t]):.2f}")
    print(f"6-Month Loan (j_{t}): {pulp.value(j[t]):.2f}")
    print(f"Quarterly Loan (k_{t}): {pulp.value(k[t]):.2f}")
print(f"\nTotal Money at Q9 (x_9): {pulp.value(x[9]):.2f}")
print(f"Final Return (r): {pulp.value(r):.2f}")