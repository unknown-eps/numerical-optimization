#Define a class that generates data for regression
#Use scikit-learn make_regression function

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class RegresssionDataMaker:
    def __init__(self,n_samples,n_features=1,noise=0.1,seed=42,true_coef=np.array([1,2])):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.seed = seed
        self.true_coef = true_coef
    def make_data(self):
        # np.random.seed(self.seed)
        X,y,coefs=make_regression(n_samples=self.n_samples,n_features=self.n_features,noise=self.noise,coef=True,random_state=self.seed)
        #Append the bias
        coefs=self.true_coef
        y=X@coefs[1:]+coefs[0]
        y=y.reshape(-1,1)
        return X,y,coefs
    # Save the generated data to a csv file
    def save_data(self,X,y,filename):
        np.savetxt(filename,np.column_stack((X,y)),delimiter=',')
        print('Data saved to',filename)
    #Save the coefficients to a csv file
    def save_coefs(self,coefs,filename):
        np.savetxt(filename,coefs,delimiter=',')
        print('Coefficients saved to',filename)    
    
    #PLot the generated data
    def plot_data(self,X,y):
        plt.scatter(X,y,color='black')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()
    # make a function to create regression data with ones
    def make_data_with_ones(self):
        X,y,coefs = self.make_data()
        X = np.hstack((np.ones((self.n_samples,1)),X))
        return X,y,coefs
    
