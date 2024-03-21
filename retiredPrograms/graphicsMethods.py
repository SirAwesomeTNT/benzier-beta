import numpy as np
import matplotlib.pyplot as plt

def plotSamplesAndBezierCurves(samples, xControlPoints, yControlPoints):
    """
    Plot Bezier curves along with control points and sample points.
    
    Args:
    - samples: Array of sample data points
    - xControlPoints: List of arrays containing x-coordinate control points for each Bezier curve
    - yControlPoints: List of arrays containing y-coordinate control points for each Bezier curve
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a list of colors for the Bezier curves
    curve_colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    # Plot Bezier curves
    for i, (xCtrl, yCtrl) in enumerate(zip(xControlPoints, yControlPoints)):
        curve_color = curve_colors[i % len(curve_colors)]  # Cycle through the list of colors
        
        # Plot thin grey lines connecting control points
        for j in range(len(xCtrl) - 1):
            ax.plot(xCtrl[j:j+2], yCtrl[j:j+2], color='lightgray', linestyle='--')

        # Plot Bézier curve
        tValues = np.linspace(0, 1, 100)
        bezierCurve = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
        fitCurve = np.dot(bezierCurve, np.array([xCtrl, yCtrl]).T)
        ax.plot(fitCurve[:, 0], fitCurve[:, 1], color=curve_color, label=f'Bezier Curve {i+1}')  # Plot Bézier curve
        
        # Plot control points in the same color as the Bezier curve
        ax.scatter(xCtrl, yCtrl, color=curve_color, label=f'Control Points {i+1}')

    # Plot sample points as black dots
    ax.scatter(np.arange(len(samples)), samples, color='black', label='Samples')
    
    ax.set_title('Bezier Curves with Sample Points')
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.grid(True)

    plt.tight_layout()
    plt.show()