import numpy as np

# Example coefficients for the linear model (y = ax + b)
a, b = 2, 1  # Replace with actual values
coefficients = np.array([a, b])

# Example data (Replace with actual data points)
x_data = np.array([[1, 1], [2, 1], [3, 1]])  # Add a column of ones for the intercept
y_data = np.array([3, 5, 7])  # Actual y values

# Predict y using matrix multiplication
y_pred = x_data @ coefficients

# Calculate the loss (Mean Squared Error)
loss = np.mean((y_data - y_pred) ** 2)

print("Predicted y:", y_pred)
print("Loss (MSE):", loss)