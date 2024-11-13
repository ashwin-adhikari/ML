import sympy as sp
from sympy import *


x = sp.symbols('x')
function = x**2

indefinite_integral = function.integrate(x)
print(indefinite_integral)

a,b =1,3
definite_integral = sp.integrate(function,(x,a,b))
print(definite_integral)


# VISUALIZATION

import numpy as np
import matplotlib.pyplot as plt

func_lambda = sp.lambdify(x,function)
#sp.lambdify(x, function) converts the symbolic function f(x)=x^2 into a function func_lambda, which can be used to calculate numerical values for specific inputs of x. This enables plotting f(x) numerically over a range of x values.
integral_lambda = sp.lambdify(x,indefinite_integral)

# generate x values
x_vals = np.linspace(0,4,100)
# linspace generates a specified number of equally spaced values between two endpoints.
y_vals = func_lambda(x_vals)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = x^2', color='blue')
plt.fill_between(x_vals, y_vals, where=[(x_val >= a and x_val <= b) for x_val in x_vals], color='gray', alpha=0.5)
plt.title('Function and Area Under Curve')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()