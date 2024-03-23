import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

t, P2 = sp.symbols('t P2')
equationX = 2 * P2 * t + 5

t_values = np.linspace(1, 10, 100)

t = t_values
print(equationX)
xLambda = lambda t: equationX
print(xLambda)


# Define the parameter values (t)
t = np.linspace(0, 2*np.pi, 100)

# Define parametric equations for x and y coordinates
equation1 = lambda t: np.cos(t)
equation2 = lambda t: np.sin(t)

# Calculate x and y coordinates using the parametric equations
x = xLambda(t_values)
y = equation2(t)

# Plot the parametric curve
plt.plot(x, y)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parametric Plot')

# Show plot
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.show()
