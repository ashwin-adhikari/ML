import sympy as sp
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

x = sp.symbols('x')
function = x**2

derivative = function.diff(x)
print(f"Derivative is: {derivative}")

# rate = lambdify(x,function)
value_at_x = 2
print(derivative.subs(x,value_at_x))
rate = lambdify(x,derivative)
print(rate(2))


# Define the function and derivative as lambdas
func_lambda = sp.lambdify(x, function, 'numpy')
deriv_lambda = sp.lambdify(x, derivative, 'numpy')

# Generate x values
x_vals = np.linspace(-5, 5, 100)
y_vals = func_lambda(x_vals)
tangent_line = deriv_lambda(value_at_x) * (x_vals - value_at_x) + func_lambda(value_at_x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = x^2')
plt.plot(x_vals, tangent_line, label='Tangent at x = 2', linestyle='dashed')
plt.scatter(value_at_x, func_lambda(value_at_x), color='red') # point of tangency
plt.title('Function and Tangent Line')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show() 