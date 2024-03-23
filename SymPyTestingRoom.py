import numpy as np
import sympy as sp

# Define symbols
ta, tb, t = sp.symbols('ta tb t')

# Define the equation
equation = sp.Eq((5*ta**3 - 5*ta**2 + ta - 1/3)/(ta**2*(ta - 1)), (5*tb**3 - 5*tb**2 + tb - 2/3)/(tb**2*(tb - 1)))

# Define grid search parameters
num_points = 50
ta_values_np = np.linspace(0.01, 0.99, num_points)
tb_values_np = np.linspace(0.01, 0.99, num_points)

# Convert numpy values to sympy symbols
ta_values = [sp.Rational(val) for val in ta_values_np]
tb_values = [sp.Rational(val) for val in tb_values_np]

# Perform grid search
valid_solutions = []
tolerance = 1e-2  # Adjust tolerance as needed
for ta_val in ta_values:
    for tb_val in tb_values:
        # Evaluate equation at (ta, tb)
        lhs = ((5*ta_val**3 - 5*ta_val**2 + ta_val - 1/3)/(ta_val**2*(ta_val - 1)))
        rhs = ((5*tb_val**3 - 5*tb_val**2 + tb_val - 2/3)/(tb_val**2*(tb_val - 1)))
        equation_difference = lhs - rhs  # Compute absolute difference
        # Check if solution satisfies desired tolerance
        if abs(equation_difference) < tolerance:
            valid_solutions.append((ta_val, tb_val))

print("Valid solutions within tolerance:")
for solution in valid_solutions:
    print(f"ta = {solution[0]}, tb = {solution[1]}")
