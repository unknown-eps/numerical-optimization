import numpy as np
## Constants
INTEREST_RATE = 0.005  # Interest rate for investment
K_REPAY_FACTOR = 1.025  # Repayment factor for 4-month loan
J_REPAY_FACTOR = (1+0.018)**2  # Repayment factor for 6-month loan
L_REPAY_FACTOR = (1+0.01)**8  # Repayment factor for 2-year loan
cash_flow = [100, 500, 100, -600, -500, 200, 600, -900]

# Decision Variables
# L: Two-year loan available at the beginning of the first quarter
# i_t: The amount invested in quarter t. t = 1 to 8
# x_t: Total money at the start of quarter t after all loans have also been taken. t = 1 to 9
# j_t: 6-month loan available at the beginning of each quarter t, t = 1 to 7
# k_t: 4-month loan available at the beginning of each quarter t, t = 1 to 8

# Total decision variables = 1 (L) + + 9 (x_t)+ 8 (i_t) + 8 (j_t) + 8 (k_t)  = 34
# We already have non-negativity constraints for all decision variables.
# The objective function is  x_9 - L_REPAY_FACTOR * L
