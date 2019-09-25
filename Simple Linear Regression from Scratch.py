# # SIMPLE LINEAR REGRESSION 
#   FROM SCRATCH
#   USING PYTHON

# Using the dataset -- Years of Experience V/S Salary


# importing Libraries
import numpy as np      # for working with arrays
import pandas as pd     # for dealing with DataFrames
import matplotlib.pyplot as plt     # for Data Visualisation




# loading the Training Dataset into Pandas.DataFrame
df = pd.read_csv("G://machine learning//SalaryData_Train.csv")
print(type(df))


df.head(10) # prints the first 10 Rows from the dataset



# Printing the Shape of Dataset
print(df.shape)





# Visualising the data
plt.plot(df["years"],df["salary"],"o")   # creates a Scatter Plot
plt.xlabel("Years of Experience")
plt.ylabel("Salary (rs)")
plt.legend(["Data Points"])
plt.title("Years of Experience VS Salary Plot")
plt.grid()
plt.show()





x = df["years"]  # independent variable
y = df['salary']  # dependent variable

x_mean = np.mean(x)
y_mean = np.mean(y)

n = len(x)   # number of rows in the dataset

# Regression Formula
# Open this link  --  https://i.stack.imgur.com/lYevl.gif
# Image Source -- Stack Exchange
num = 0
den = 0
for i in range(n):
        num += (x[i]-x_mean)*(y[i]-y_mean)
        den += (x[i]-x_mean)**2
        
# m --> slope
# c --> y-intercept
# General Equation of Line --> y = m*x + c

m = num/den
c = y_mean - m*x_mean

print ("m = ",m,'\t',"c = ",c)
print("\nThe Equation of the Regression Line generated is : \n")
print("y = ", m, "*","x", " + ", c)




y_pred = np.array(m*x+c)
y_pred





# Data Visualisation
plt.scatter(x,y_pred,color='r')  
plt.scatter(x,y,color='g')
plt.plot(x,y_pred,color='b')
plt.xlabel("Years of Experience")
plt.ylabel("Salary (Rs)")
plt.title("Years of Experience V/S Salary")
plt.legend(["Regression Line","Predicted Data Points","Data Points"])
plt.grid()
plt.show()





# Calculating the Root Mean Squared Error (RMSE)
# Gives the Idea about : 
#  -- How far from the regression line data points are
#  -- It tells you how concentrated the data is around the line of best fit.
numerator = 0
denominator = 0
for i in range(n):
    numerator += (y_pred[i] - y_mean)**2
    denominator += (y[i] - y_mean)**2
    
rmse = numerator/denominator
print ("\nRoot Mean Squared Error = ",rmse)





# Loading the Testing Dataset
df_test = pd.read_csv("G:/machine learning/SalaryData_Test.csv")

# The column for Salary will be empty 
# i.e. in Python, it will be having null value.
print("\n",df_test)





# Using the Regression Line for predicting the Test-Data values
test_x = df_test["years"]
test_y = df_test["salary"]

test_y = m*test_x + c





# Visualising the Data Graphically
plt.plot(x,y_pred, color='b')
plt.scatter(test_x,test_y,color='r')





# Printing the Predicted Values
df_test['salary'] = test_y
print("\n",df_test)


# ### Made By : Karansinh Padhiar
