# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Estimate coefficients(Find m and c in equation y = mx +c with the input data x to predict y in future).
def estimate_coefficients(x, y):
    # ss_xy = summation of (x * y) - n * mean of x * mean of y.
    # ss_xx = summation of square of x - n * square of mean of x * square.
    # b1 = ss_xx / ss_xx which gives the slope of our regression(value of m).
    # b0 = mean of y - n * mean of x which gives the y intercept(value of c).

    # No of data points.
    n = np.size(x)

    # Mean of x and y
    mx = np.mean(x)
    my = np.mean(y)

    # Cross deviation about x
    ss_xy = np.sum(x * y) - n * mx * my
    ss_xx = np.sum(x * x) - n * mx * mx

    # Calculating regression coefficient
    b1 = ss_xy / ss_xx
    b0 = my - b1 * mx 

    return (b0, b1) # return value of c and m

# Prediction (find value of y with x and estimated coefficients c and m).
def predict(x, b):
    return np.round(b[1] * x + b[0])

# Visualization of regression.
def plot_regression(x, y, b):
    # Plot data points in scatter plot.
    plt.scatter(x, y, s=30)

    # Prediction y
    y_pred = predict(x, b)

    # Plot predictid regression line 
    plt.plot(x, y_pred)
    plt.show()

# Read data 
data = pd.read_csv("Salary_Data.csv")
x=data['YearsExperience'].to_numpy()
y=data['Salary'].to_numpy()

print("================================")
# estimating coefficients
b = estimate_coefficients(x, y)

print("Estimated coefficients: \n ~~~~~~~~~~~~~~~~~~~~~\nb0 = {} \nb1 = {}".format(b[0], b[1]))
print("________________________________")
# prediction
print("Twenty Years experiance of salary will get salary of: ",predict(20, b))

print("================================")
# plotting regression line
plot_regression(x, y, b)
