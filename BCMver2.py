import numpy as np

def calculateControlPoints(xSamples, ySamples):
    """
    Calculate the control points of a Bezier curve based on given sample points.
    
    Args:
    - xSamples: Array of original x-coordinate sample points
    - ySamples: Array of original y-coordinate sample points
    
    Returns:
    - xCtrl: Array of x-coordinate control points
    - yCtrl: Array of y-coordinate control points
    """
    # Reshape arrays vertically
    xSamples = xSamples.reshape(-1, 1)
    ySamples = ySamples.reshape(-1, 1)

    # Coefficients array
    coefficients = np.array([[6, 0, 0, 0], [-5, 18, -9, 2], [2, -9, 18, -5], [0, 0, 0, 6]])

    # Calculate control points
    xCtrl = coefficients @ xSamples / 6
    yCtrl = coefficients @ ySamples / 6

    # Flatten arrays horizontally
    xCtrl = xCtrl.reshape(1, -1)
    yCtrl = yCtrl.reshape(1, -1)

    return xCtrl, yCtrl
