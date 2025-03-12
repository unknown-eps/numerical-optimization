[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HuZYJ5y3)
# Mid Term Exam

Coal-fired power plants convert the chemical energy in coal to electrical energy through combustion. The efficiency of this conversion depends on various operational parameters, including the coal-to-air ratio, combustion temperature, steam pressure and coal properties (sulphur content, water content). Finding the optimal values for these parameters can significantly improve plant efficiency and reduce operational costs.

Part 1: Use the provided data of coal-to-air ratio, combustion temperature, steam pressure, coal properties and corresponding efficiency to develop an explicit function of efficiency in terms of other operational parameters. From literature fed to Claude Sonnet 3.7, it is known that the relationship between efficiency and these variables are quadratic. 
1.	Use batch gradient descent, and mini-batch gradient descent and mini-batch ADAM to find the functional form of efficiency
2.	Visualize the objective function and the solution path for all three.
3.	Explain the convergence behaviour for different batch sizes in the two mini-batch algorithms

Part 2: Set up an unconstrained optimization problem to find the optimal operational parameters from the function you found above. Prepare your code in such a way that, the end user can choose any of the solutions from Part 1. 
1.	Use Newton with exact line search, BFGS and Conjugate gradient for solving the unconstrained optimization problem
2.	Visualize the objective function and the solution path
3.	Explain the convergence behaviour. 

Note: There are different visualizations possible, be creative about how to explain the maximum information in minimum plots.
