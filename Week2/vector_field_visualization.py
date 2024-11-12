# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create a grid
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 20, 10)
X, Y = np.meshgrid(x, y)

# Step 3: Define the vector field
# U and V are the components of the vector
U = -Y  # Negative Y component for circular pattern
V = X   # X component for circular pattern

# Step 4: Plot the vector field
plt.quiver(X, Y, U, V)

# Step 5: Formatting the plot
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title("Circular Vector Field")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid()

# Step 6: Display the plot
plt.show()