#!/usr/bin/env python
# coding: utf-8

# In[523]:


#Michael Lust: 801094861
#ECGR 4105 Intro to Machine Learning
#September 30, 2021
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[524]:


housing = pd.DataFrame(pd.read_csv('Housing.csv'))
housing.head()                      # To get first n rows from the dataset default value of n is 5


# In[525]:


M = len(housing)
M


# In[526]:


housing.shape


# In[527]:


housing.info()


# In[528]:


housing.describe()


# In[529]:


# You can see that your dataset has many columns with values as 'Yes' or 'No'.
# But in order to fit a regression line, we would need numerical values and not string.
# List of variables to map

varlist=['mainroad','guestroom','basement','hotwaterheating','airconditioning', 'prefarea']

#Defining the map function
def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

#Applying the function of the housing list
housing[varlist] = housing[varlist].apply(binary_map)

#Check the housing dataframe
housing.head()


# In[530]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows,
np.random.seed(0)
df_train,df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 1)
df_train.shape


# In[531]:


df_test.shape


# In[532]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[533]:


df_Newtest.head()


# In[534]:


df_Newtrain.shape


# In[535]:


df_Newtest.shape


# In[536]:


#Problem 1


# In[537]:


#-------------------------------------------------------------------------------------------------------------------------#


# In[538]:


#Starting of with testing the training set at 70%
area_T = df_Newtrain.values[:, 0] #Getting the values of each variable in the first column
bedrooms_T = df_Newtrain.values[:, 1] #Getting the values of each variable in the second column
bathrooms_T = df_Newtrain.values[:, 2] #Getting the values of each variable in the third column
stories_T = df_Newtrain.values[:, 3] #Getting the values of each variable in the fourth column
mainroad_T = df_Newtrain.values[:, 4] #Getting the values of each variable in the fifth column
guestroom_T = df_Newtrain.values[:, 5] #Getting the values of each variable in the first column
basement_T = df_Newtrain.values[:, 6] #Getting the values of each variable in the first column
hotwaterheating_T = df_Newtrain.values[:, 7] #Getting the values of each variable in the first column
airconditioning_T = df_Newtrain.values[:, 8] #Getting the values of each variable in the first column
parking_T = df_Newtrain.values[:, 9] #Getting the values of each variable in the first column
prefarea_T = df_Newtrain.values[:, 10] #Getting the values of each variable in the first column
price_T = df_Newtrain.values[:, 11] #Setting the last column as my result for y = price.

#Then continuting with testing the validation set at 30%
area_V = df_Newtest.values[:, 0] #Getting the values of each variable in the first column
bedrooms_V = df_Newtest.values[:, 1] #Getting the values of each variable in the second column
bathrooms_V = df_Newtest.values[:, 2] #Getting the values of each variable in the third column
stories_V = df_Newtest.values[:, 3] #Getting the values of each variable in the fourth column
mainroad_V = df_Newtest.values[:, 4] #Getting the values of each variable in the fifth column
guestroom_V = df_Newtest.values[:, 5] #Getting the values of each variable in the first column
basement_V = df_Newtest.values[:, 6] #Getting the values of each variable in the first column
hotwaterheating_V = df_Newtest.values[:, 7] #Getting the values of each variable in the first column
airconditioning_V = df_Newtest.values[:, 8] #Getting the values of each variable in the first column
parking_V = df_Newtest.values[:, 9] #Getting the values of each variable in the first column
prefarea_V = df_Newtest.values[:, 10] #Getting the values of each variable in the first column
price_V = df_Newtest.values[:, 11] #Setting the last column as my result for y = price.

M_T = len(price_T) #Number of training examples
M_V = len(price_V) #Number of validation examples

print('M = ', M)


# In[539]:


#Printing the training set at 70%
print('X = ', area_T[: 381]) # Show all the data points for X1
print('X = ', bedrooms_T[: 381]) # Show all the data points for X2
print('X = ', bathrooms_T[: 381]) # Show all the data points for X3
print('X = ', stories_T[: 381]) # Show all the data points for X4
print('X = ', mainroad_T[: 381]) # Show all the data points for X5
print('X = ', guestroom_T[: 381]) # Show all the data points for X6
print('X = ', basement_T[: 381]) # Show all the data points for X7
print('X = ', hotwaterheating_T[: 381]) # Show all the data points for X8
print('X = ', airconditioning_T[: 381]) # Show all the data points for X9
print('X = ', parking_T[: 381]) # Show all the data points for X10
print('X = ', prefarea_T[: 381]) # Show all the data points for X11
print('Y = ', price_T[: 381]) # Show all the data points for Y

print('M = ', M_T)


# In[540]:


#Printing the validation set at 30%
print('X = ', area_V[: 164]) # Show all the data points for X1
print('X = ', bedrooms_V[: 164]) # Show all the data points for X2
print('X = ', bathrooms_V[: 164]) # Show all the data points for X3
print('X = ', stories_V[: 164]) # Show all the data points for X4
print('X = ', mainroad_V[: 164]) # Show all the data points for X5
print('X = ', guestroom_V[: 164]) # Show all the data points for X6
print('X = ', basement_V[: 164]) # Show all the data points for X7
print('X = ', hotwaterheating_V[: 164]) # Show all the data points for X8
print('X = ', airconditioning_V[: 164]) # Show all the data points for X9
print('X = ', parking_V[: 164]) # Show all the data points for X10
print('X = ', prefarea_V[: 164]) # Show all the data points for X11
print('Y = ', price_V[: 164]) # Show all the data points for Y

print('M = ', M_V)


# In[541]:


def calculate_scalar(X, Y, theta): #Declaring values and computing the Scalar value J
    
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J


# In[542]:


def gradient_descent(X, Y, theta, alpha, iterations):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = (alpha/M)*X.transpose().dot(errors); #learning rate over training examples * scalar of resulting dot product.  
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[543]:


# Part 1-a


# In[544]:


# Using reshape function convert all X variables 1D array to 2D array for Training Set
X_T = np.ones((M_T,1))
Y_T = price_T 

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,parking_T))


# In[545]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01; #This is to avoid getting overfill error.
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[546]:


#Calculating gradient descent with theta and scalar J for training set
theta, result_T, theta_interval = gradient_descent(X_T, Y_T, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[547]:


# Using reshape function convert all X variables 1D array to 2D array for Validation Set
X_V = np.ones((M_V,1))
Y_V = price_V 

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,parking_V))


# In[548]:


#Making a theta array with initializations of O and setting validation parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01; #This is to avoid getting overfill error.;
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[549]:


#Calculating gradient descent with theta and scalar J for validation set
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[550]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[551]:


#Part 1-b


# In[552]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_T = np.ones((M_T,1))
Y_T = price_T

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
mainroad_T = mainroad_T.reshape(M_T,1)
guestroom_T = guestroom_T.reshape(M_T,1)
basement_T = basement_T.reshape(M_T,1)
hotwaterheating_T = hotwaterheating_T.reshape(M_T,1)
airconditioning_T = airconditioning_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)
prefarea_T = prefarea_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,mainroad_T,guestroom_T,basement_T,hotwaterheating_T,airconditioning_T,parking_T,prefarea_T))


# In[553]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01; #This is to avoid getting overfill error.
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[554]:


#Calculating gradient descent with theta and scalar J
theta, result_T, theta_interval = gradient_descent(X_T, Y_T, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[555]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_V = np.ones((M_V,1))
Y_V = price_V

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
mainroad_V = mainroad_V.reshape(M_V,1)
guestroom_V = guestroom_V.reshape(M_V,1)
basement_V = basement_V.reshape(M_V,1)
hotwaterheating_V = hotwaterheating_V.reshape(M_V,1)
airconditioning_V = airconditioning_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)
prefarea_V = prefarea_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,mainroad_V,guestroom_V,basement_V,hotwaterheating_V,airconditioning_V,parking_V,prefarea_V))


# In[556]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01; #This is to avoid getting overfill error.
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[557]:


#Calculating gradient descent with theta and scalar J
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[558]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[559]:


#*************************************************************************************************************************#
#*************************************************************************************************************************#


# In[560]:


#Problem 2


# In[561]:


#-------------------------------------------------------------------------------------------------------------------------#


# In[562]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[563]:


df_Newtest.head()


# In[564]:


df_Newtrain.shape


# In[565]:


df_Newtest.shape


# In[ ]:





# In[566]:


#Here we can see that except for area, all the columns have small integer values.
#So it is extremely important to rescale the variables so that they have a comparable scales
#If we don't have comparable scales, then some of the coefficients as obtained by fitting
#This might become very annoying at the time of model evaluation. 
#So it is advised to use standardization or normalization so that the units of the coefficient 

#As you know, there are two common ways of rescaling:
#1. Min-Max scaling
#2. Standardisation (mean-0, sigma-1)

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#define standard scalar
#scalar = StandardScalar()
scalar = MinMaxScaler()
df_Newtrain[num_vars] = scalar.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[567]:


df_Newtest[num_vars] = scalar.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[568]:


#Starting of with testing the training set at 70%
area_T = df_Newtrain.values[:, 0] #Getting the values of each variable in the first column
bedrooms_T = df_Newtrain.values[:, 1] #Getting the values of each variable in the second column
bathrooms_T = df_Newtrain.values[:, 2] #Getting the values of each variable in the third column
stories_T = df_Newtrain.values[:, 3] #Getting the values of each variable in the fourth column
mainroad_T = df_Newtrain.values[:, 4] #Getting the values of each variable in the fifth column
guestroom_T = df_Newtrain.values[:, 5] #Getting the values of each variable in the first column
basement_T = df_Newtrain.values[:, 6] #Getting the values of each variable in the first column
hotwaterheating_T = df_Newtrain.values[:, 7] #Getting the values of each variable in the first column
airconditioning_T = df_Newtrain.values[:, 8] #Getting the values of each variable in the first column
parking_T = df_Newtrain.values[:, 9] #Getting the values of each variable in the first column
prefarea_T = df_Newtrain.values[:, 10] #Getting the values of each variable in the first column
price_T = df_Newtrain.values[:, 11] #Setting the last column as my result for y = price.

#Then continuting with testing the validation set at 30%
area_V = df_Newtest.values[:, 0] #Getting the values of each variable in the first column
bedrooms_V = df_Newtest.values[:, 1] #Getting the values of each variable in the second column
bathrooms_V = df_Newtest.values[:, 2] #Getting the values of each variable in the third column
stories_V = df_Newtest.values[:, 3] #Getting the values of each variable in the fourth column
mainroad_V = df_Newtest.values[:, 4] #Getting the values of each variable in the fifth column
guestroom_V = df_Newtest.values[:, 5] #Getting the values of each variable in the first column
basement_V = df_Newtest.values[:, 6] #Getting the values of each variable in the first column
hotwaterheating_V = df_Newtest.values[:, 7] #Getting the values of each variable in the first column
airconditioning_V = df_Newtest.values[:, 8] #Getting the values of each variable in the first column
parking_V = df_Newtest.values[:, 9] #Getting the values of each variable in the first column
prefarea_V = df_Newtest.values[:, 10] #Getting the values of each variable in the first column
price_V = df_Newtest.values[:, 11] #Setting the last column as my result for y = price.

M_T = len(price_T) #Number of training examples
M_V = len(price_V) #Number of validation examples

print('M = ', M)


# In[569]:


#Printing the training set at 70%
print('X = ', area_T[: 381]) # Show all the data points for X1
print('X = ', bedrooms_T[: 381]) # Show all the data points for X2
print('X = ', bathrooms_T[: 381]) # Show all the data points for X3
print('X = ', stories_T[: 381]) # Show all the data points for X4
print('X = ', mainroad_T[: 381]) # Show all the data points for X5
print('X = ', guestroom_T[: 381]) # Show all the data points for X6
print('X = ', basement_T[: 381]) # Show all the data points for X7
print('X = ', hotwaterheating_T[: 381]) # Show all the data points for X8
print('X = ', airconditioning_T[: 381]) # Show all the data points for X9
print('X = ', parking_T[: 381]) # Show all the data points for X10
print('X = ', prefarea_T[: 381]) # Show all the data points for X11
print('Y = ', price_T[: 381]) # Show all the data points for Y

print('M = ', M_T)


# In[570]:


#Printing the validation set at 30%
print('X = ', area_V[: 164]) # Show all the data points for X1
print('X = ', bedrooms_V[: 164]) # Show all the data points for X2
print('X = ', bathrooms_V[: 164]) # Show all the data points for X3
print('X = ', stories_V[: 164]) # Show all the data points for X4
print('X = ', mainroad_V[: 164]) # Show all the data points for X5
print('X = ', guestroom_V[: 164]) # Show all the data points for X6
print('X = ', basement_V[: 164]) # Show all the data points for X7
print('X = ', hotwaterheating_V[: 164]) # Show all the data points for X8
print('X = ', airconditioning_V[: 164]) # Show all the data points for X9
print('X = ', parking_V[: 164]) # Show all the data points for X10
print('X = ', prefarea_V[: 164]) # Show all the data points for X11
print('Y = ', price_V[: 164]) # Show all the data points for Y

print('M = ', M_V)


# In[571]:


# Part 2-a


# In[572]:


# Using reshape function convert all X variables 1D array to 2D array for Training Set
X_T = np.ones((M_T,1))
Y_T = price_T 

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,parking_T))


# In[573]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01;
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[574]:


#Calculating gradient descent with theta and scalar J for training set
theta, result_T, theta_interval = gradient_descent(X_T, Y_T, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[575]:


# Using reshape function convert all X variables 1D array to 2D array for Validation Set
X_V = np.ones((M_V,1))
Y_V = price_V 

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,parking_V))


# In[576]:


#Making a theta array with initializations of O and setting validation parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01;
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[577]:


#Calculating gradient descent with theta and scalar J for validation set
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[578]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[579]:


#Part 2-b


# In[580]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_T = np.ones((M_T,1))
Y_T = price_T

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
mainroad_T = mainroad_T.reshape(M_T,1)
guestroom_T = guestroom_T.reshape(M_T,1)
basement_T = basement_T.reshape(M_T,1)
hotwaterheating_T = hotwaterheating_T.reshape(M_T,1)
airconditioning_T = airconditioning_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)
prefarea_T = prefarea_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,mainroad_T,guestroom_T,basement_T,hotwaterheating_T,airconditioning_T,parking_T,prefarea_T))


# In[581]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01;
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[582]:


#Calculating gradient descent with theta and scalar J
theta, result_T, theta_interval = gradient_descent(X_T, Y_T, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[583]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_V = np.ones((M_V,1))
Y_V = price_V

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
mainroad_V = mainroad_V.reshape(M_V,1)
guestroom_V = guestroom_V.reshape(M_V,1)
basement_V = basement_V.reshape(M_V,1)
hotwaterheating_V = hotwaterheating_V.reshape(M_V,1)
airconditioning_V = airconditioning_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)
prefarea_V = prefarea_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,mainroad_V,guestroom_V,basement_V,hotwaterheating_V,airconditioning_V,parking_V,prefarea_V))


# In[584]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01;
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[585]:


#Calculating gradient descent with theta and scalar J
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[586]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[587]:


#*************************************************************************************************************************#
#*************************************************************************************************************************#


# In[588]:


#Problem 3


# In[589]:


#-------------------------------------------------------------------------------------------------------------------------#


# In[590]:


df_Newtrain.head(20)


# In[591]:


df_Newtest.head(20)


# In[592]:


#Illistrating the changes needed for adding regularization parameterpenalties
def calculate_scalar(X, Y, theta): #Declaring values and computing the Scalar value J
    
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J


# In[593]:


#Need to modify the gradient decent logic for your training set only
def gradient_descent_training(X, Y, theta, alpha, delta, iterations):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = alpha*((1/M)*X.transpose().dot(errors) + ((delta/M)*theta)); #learning rate over training examples * scalar 
                                                                               #of resulting dot product with parameter penalties
                                                                               #for Regularization.
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[594]:


# Part 3-a


# In[595]:


# Using reshape function convert all X variables 1D array to 2D array for Training Set
X_T = np.ones((M_T,1))
Y_T = price_T 

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,parking_T))


# In[596]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01;
delta = 10;
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[597]:


#Calculating gradient descent with theta and scalar J for training set
theta, result_T, theta_interval = gradient_descent_training(X_T, Y_T, theta, alpha, delta, iterations) #Changed for Regularization
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[598]:


# Using reshape function convert all X variables 1D array to 2D array for Validation Set
X_V = np.ones((M_V,1))
Y_V = price_V 

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,parking_V))


# In[599]:


#Making a theta array with initializations of O and setting validation parameters.
theta = np.zeros(6)
iterations = 500;
alpha = 0.01;
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[600]:


#Calculating gradient descent with theta and scalar J for validation set
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[601]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[602]:


#Part 3-b


# In[603]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_T = np.ones((M_T,1))
Y_T = price_T

area_T = area_T.reshape(M_T,1)
bedrooms_T = bedrooms_T.reshape(M_T,1)
bathrooms_T = bathrooms_T.reshape(M_T,1)
stories_T = stories_T.reshape(M_T,1)
mainroad_T = mainroad_T.reshape(M_T,1)
guestroom_T = guestroom_T.reshape(M_T,1)
basement_T = basement_T.reshape(M_T,1)
hotwaterheating_T = hotwaterheating_T.reshape(M_T,1)
airconditioning_T = airconditioning_T.reshape(M_T,1)
parking_T = parking_T.reshape(M_T,1)
prefarea_T = prefarea_T.reshape(M_T,1)

X_T = np.hstack((X_T,area_T,bedrooms_T,bathrooms_T,stories_T,mainroad_T,guestroom_T,basement_T,hotwaterheating_T,airconditioning_T,parking_T,prefarea_T))


# In[604]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01;
delta = 100;
result_T = calculate_scalar(X_T,Y_T, theta)
print('Scalar values is ', result_T) #Print the scalar value for Gradient Descent


# In[605]:


#Calculating gradient descent with theta and scalar J
theta, result_T, theta_interval = gradient_descent_training(X_T, Y_T, theta, alpha, delta, iterations)
print('Final value of theta =', theta)
print('Y = ', result_T)


# In[606]:


# Using reshape function convert all X variables 1D array to 2D array for training set
X_V = np.ones((M_V,1))
Y_V = price_V

area_V = area_V.reshape(M_V,1)
bedrooms_V = bedrooms_V.reshape(M_V,1)
bathrooms_V = bathrooms_V.reshape(M_V,1)
stories_V = stories_V.reshape(M_V,1)
mainroad_V = mainroad_V.reshape(M_V,1)
guestroom_V = guestroom_V.reshape(M_V,1)
basement_V = basement_V.reshape(M_V,1)
hotwaterheating_V = hotwaterheating_V.reshape(M_V,1)
airconditioning_V = airconditioning_V.reshape(M_V,1)
parking_V = parking_V.reshape(M_V,1)
prefarea_V = prefarea_V.reshape(M_V,1)

X_V = np.hstack((X_V,area_V,bedrooms_V,bathrooms_V,stories_V,mainroad_V,guestroom_V,basement_V,hotwaterheating_V,airconditioning_V,parking_V,prefarea_V))


# In[607]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(12)
iterations = 500;
alpha = 0.01;
result_V = calculate_scalar(X_V,Y_V, theta)
print('Scalar values is ', result_V) #Print the scalar value for Gradient Descent


# In[608]:


#Calculating gradient descent with theta and scalar J
theta, result_V, theta_interval = gradient_descent(X_V, Y_V, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result_V)


# In[609]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_T, color='Red', label='Training' )
plt.plot(range(1, iterations + 1), result_V, color='Blue', label='Validation' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J for Training and Validation')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()

