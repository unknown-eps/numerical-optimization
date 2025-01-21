import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("real_estate_dataset.csv")
print(df.columns)
n_samples,n_features=df.shape
columns=df.columns
np.savetxt("column_names.csv",columns,fmt="%s")

# Use square_feet, Garage_Size, Location_Score, Distance_to_Center as features

X=df[["Square_Feet","Garage_Size","Location_Score","Distance_to_Center"]]

y=df["Price"].values

print(f"Shape of X: {X.shape} \n")
#Print data typer of X
print(f"Data type of X:{X.dtypes} \n")
# get the number of samples and fatures in X
n_samples,n_features=X.shape
#Build a linear model
coefs=np.ones(n_features+1)

# predict the price for each sample in X

predictions_bydefn = X @ coefs[1:] +coefs[0]

# Trick to append a column of 1's to X
X=np.hstack((np.ones((n_samples,1)),X))
predictions = X @ coefs

is_same=np.allclose(predictions_bydefn,predictions)

print("Are the predictions the same?",is_same)

#Calculate the error using predictions and y
errors=y-predictions
# print the size of errors and it's L2 norm
rel_errors=errors/y
print(f"Size of errors:{errors.shape}")
print(f"L2 norm errors:{np.linalg.norm(errors)}")
print(f"L2 norm of relative error:{np.linalg.norm(rel_errors)}")
loss_loop = 0
for i in range(n_samples):
    loss_loop = loss_loop +errors[i]**2
loss_loop= loss_loop/n_samples

# calculate the mean of square of errors using matrix operations
loss_matrix=np.transpose(errors)@ errors/n_samples
is_diff = np.allclose(loss_loop,loss_matrix)
# Are the errors same
print("Are the errors the same?",is_diff)
print(f"Size of errors:{errors.shape}")
print(f"L2 norm of errors:{np.linalg.norm(rel_errors)}")
print(f"L2 norm of relative errors:{np.linalg.norm(rel_errors)}")

# What is my optimization problem

# How to find a solution?
# By searching for the coefficients at which gradient of objective function is zero
# Or I can set the gradient to zero and solve for the coefficients

# Write the loss matrix in terms of the data matrix X and the coefficients
# loss = (y-X@coefs)^T(y-X@coefs)/n_samples
loss_matrix = (y-X@coefs).T @(y-X@coefs)/n_samples
# calculate the gradient of the loss function with respect to the coefficients
grad_matrix=-2*X.T@(y-X@coefs)/n_samples
# We set grad_matrix to zero and solve for coefs
# X.T@X@coefs=X.T@y
# coefs=(X.T@X)^-1@X.T@y This if the normal equation
coefs=np.linalg.inv(X.T@X)@X.T@y
#save the coefficients to a file
np.savetxt("coefs.csv",coefs,delimiter=",")
# calculate the predictions using the new coefficients
predictions_model=X@coefs
# calculate the errors using the new coefficients
errors_model=y-predictions_model
#Print the L2 norm of the errors
print(f"L2 norm of errors_model:{np.linalg.norm(errors_model)}")
# Print the L2 norm of the relative errors model
rel_errors_model=errors_model/y
print(f"L2 norm of relative errors_model:{np.linalg.norm(rel_errors_model)}")

# Use all features in the dataset to build a linear model
X=df.drop(columns=["Price"])
y=df["Price"].values
n_samples,n_features=X.shape
print("Number of samples and features in X:",n_samples,n_features)
# solve for the coefficients using the normal equation
X=np.hstack((np.ones((n_samples,1)),X))
coefs=np.linalg.inv(X.T@X)@X.T@y
# save the coefficients to a file named coefs_all.csv
np.savetxt("coefs_all.csv",coefs,delimiter=",")

# CAlculate and print the rank of X.T @X
rank_XTX=np.linalg.matrix_rank(X.T@X)
print(f"Rank of X.T@X:{rank_XTX}")
# Solve the nomal equation usiong matrix decompositions
# QR Factorization
Q,R=np.linalg.qr(X)
print(f"Shape of Q:{Q.shape}")
print(f"Shape of R:{R.shape}")
#Write R to a file named R.csv
np.savetxt("R.csv",R,delimiter=",")

# R*coees= b
sol=Q.T@Q
#save sol to a file named sol.csv
np.savetxt("sol.csv",sol,delimiter=",")
#X=QR
#X.T@X = R.T@Q.T@Q@R = R.T@R= R.T@R
#X.T @y = R.T@Q.T@y
#R@coefs=Q.T@y
b=Q.T@y
coeffs_qr=np.linalg.inv(R)@b
#loop to solve R@coeffs = b
print(f"Shape of b:{b.shape}")
print(f"Shape of R:{R.shape}")
coeffs_qr_loop=np.zeros(n_features+1)
for i in range(n_features,-1,-1):
    coeffs_qr_loop[i]=b[i]
    for j in range(i+1,n_features+1):
        coeffs_qr_loop[i]=coeffs_qr_loop[i]-R[i,j]*coeffs_qr_loop[j]
    coeffs_qr_loop[i]=coeffs_qr_loop[i]/R[i,i]
    
# save the coeffs_qr_loop to a file named coeffs_qr_loop.csv
np.savetxt("coeffs_qr_loop.csv",coeffs_qr_loop,delimiter=",")
# Are the coefficients the same with the two methods
is_same=np.allclose(coeffs_qr,coeffs_qr_loop)
print("Are the coefficients the same qr and qr_loop?",is_same)


# Homework Part 1 via SVD
# Find the SVD of X


U,S,Vt=np.linalg.svd(X,full_matrices=False)

# X.T@X=X.T@y 
# Substitute X=U@S@Vt
# V@S.T@S@Vt@coefs=V@S@U.T@y
# coefs=V @ S^-1 @ U.T @ y
coefs_svd=Vt.T @ np.diag(1/S) @ U.T @ y

# save the coefs_svd to a file named coefs_svd.csv
np.savetxt("coefs_svd.csv",coefs_svd,delimiter=",")
# Check if the coefs_svd is the same as coefs_qr
is_same=np.allclose(coefs_svd,coeffs_qr)
print("Are the coefficients the same svd and qr?",is_same)

# Homework Part 2 via eigendecomposition of X.T@X
# X.T@X=V@Lambda@V.T
# X.T@X@coefs=b
# V@Lambda@V.T@coefs = b
# coefs=V@Lambda^-1@V.T@b
Lambda,V=np.linalg.eig(X.T@X)
coefs_eig=V @ np.diag(1/Lambda) @ V.T @ X.T @ y
# save the coefs_eig to a file named coefs_eig.csv
np.savetxt("coefs_eig.csv",coefs_eig,delimiter=",")
# Check if the coefs_eig is the same as coefs_qr
is_same=np.allclose(coefs_eig,coeffs_qr)
print("Are the coefficients the same eig and qr?",is_same)

# Vis psuedo inverse
coeffs_svd_pinv=np.linalg.pinv(X)@y
np.savetxt("coeffs_svd_pinv.csv",coeffs_svd_pinv,delimiter=",")
is_same=np.allclose(coeffs_svd_pinv,coefs_svd)
print("Are the coefficients the same svd and pinv?",is_same)

# Let's plot the data on X[:,1] vs y-axis
plt.scatter(X[:,1],y)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Price vs Square Feet")
plt.show()
plt.savefig("Price_vs_Square_Feet.png")
# Also plot a regression line with only X{:,0] and X[:,1]
# FIrst make x[:,1] as np.arange between min and max of X[:,1]
# Then calculate the predictions using the coefficients

# USe X as only square feet
X=df[["Square_Feet"]].values
y=df["Price"].values
# Add column of ones to X
X=np.hstack((np.ones((n_samples,1)),X))
coefs_1=np.linalg.inv(X.T@X)@X.T@y
X_features=np.arange(np.min(X[:,1]),np.max(X[:,1]),1)
X_features=np.hstack((np.ones((X_features.shape[0],1)),X_features.reshape(-1,1)))
plt.scatter(X[:,1],y)
plt.plot(X_features[:,1],X_features@coefs_1,color="red")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.show()
plt.savefig("Price_vs_Square_Feet_Line.png")
