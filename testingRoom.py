import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_two_arrays(array1, array2):
    if array1.shape != (2, 10) or array2.shape != (2, 10):
        raise ValueError("Arrays must have shape (2, 10)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(array1[0], array1[1], color='blue', label='Array 1', zorder=2)
    slope1, intercept1, _, _, _ = linregress(array1[0], array1[1])
    ax1.plot(array1[0], slope1 * array1[0] + intercept1, color='blue', linestyle='--', label='Line of best fit', zorder=1)
    ax1.set_title('Array 1')
    ax1.set_xlabel('X values')
    ax1.set_ylabel('X-Coefficients')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.scatter(array2[0], array2[1], color='red', label='Array 2', zorder=2)
    slope2, intercept2, _, _, _ = linregress(array2[0], array2[1])
    ax2.plot(array2[0], slope2 * array2[0] + intercept2, color='red', linestyle='--', label='Line of best fit', zorder=1)
    ax2.set_title('Array 2')
    ax2.set_xlabel('Y values')
    ax2.set_ylabel('Y-Coefficients')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Example arrays
array1 = np.array([[1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 13.0, 14.0],
                    [-0.888888888888889, -1.11111111111111, -2.88888888888889,
                    -3.11111111111111, -4.88888888888889, -5.11111111111111,
                    -6.88888888888889, -7.11111111111111, -8.88888888888889,
                    -9.11111111111111]])

array2 = np.array([[1.0, 5.0, 2.0, 6.0, 5.0, 5.0, 3.0, 5.0, 1.0, 5.0],
                    [0.185185185185185, -4.85185185185185, -2.00000000000000,
                    -6.00000000000000, -4.77777777777778, -3.22222222222222,
                    -1.14814814814815, -4.18518518518519, -0.296296296296296,
                    -4.03703703703704]])

# Plotting the arrays
plot_two_arrays(array1, array2)
