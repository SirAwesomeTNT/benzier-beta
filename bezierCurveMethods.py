import numpy as np
from math import sqrt
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Define matrix m, which contains coefficients for the cubic Bézier curve
m = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

def calculateLeastSquaresBezierControlPoints(x, y):

    s = calculateLeastSquaresCoefficientsMatrix(x, y)

    # Calculate the values for the p array that gives us the x and y locations of the final control points
    # Using the following equation, points are calculated: inv(m) * inv(s.T * s) * s.T * (y or x)
    # Actual calculations
    xP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), x)
    yP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), y)

    # Create matrix p, in which control points will be stored
    controlPoints = np.hstack((xP, yP))

    xCtrl, yCtrl = controlPoints[:, 0], controlPoints[:, 1]

    return xCtrl, yCtrl

def calculateLeastSquaresCoefficientsMatrix(x, y):
    # Calculate the values for leastSquaresCoefficients matrix, which stores the values needed for a least squares regression analysis
    # The last column is filled with ones (bezierIndex[i]^0)

    bezierIndex = calculateBezierIndexMatrix(x, y)

    leastSquaresCoefficients = np.ones((x.size, 4))
    for i in range(0, bezierIndex.size):
        leastSquaresCoefficients[i, 0] = bezierIndex[i] ** 3
        leastSquaresCoefficients[i, 1] = bezierIndex[i] ** 2
        leastSquaresCoefficients[i, 2] = bezierIndex[i] ** 1

    return leastSquaresCoefficients

def calculateBezierIndexMatrix(x, y):
    # Calculate the values for the bezierIndex matrix, which stores the respective indexes of the points on the cubic Bezier curve

    # Calculate distance matrix
    distance = calculateDistanceMatrix(x, y)

    # Initialize matrix
    bezierIndex = np.zeros(x.size)

    # This loop doesn't iterate over the first value in bezierIndex, since bezierIndex[0] = 0
    for i in range(1, x.size):
        # bezierIndex[i] = length of the most recent segment / length of all segments
        bezierIndex[i] = (distance[i]) / distance[distance.size - 1]
    
    return bezierIndex

def calculateDistanceMatrix(x, y):
    # Calculate the values for the distance matrix, which stores the distance from the start of the parent curve to each consecutive point
    distance = np.zeros(x.size)

    # This loop doesn't iterate over the first value in d since d[0] = 0
    for i in range(1, x.size):
        x1 = np.ndarray.item(x[i])
        x2 = np.ndarray.item(x[i - 1])
        y1 = np.ndarray.item(y[i])
        y2 = np.ndarray.item(y[i - 1])

        # a^2 + b^2 = c^2, solving for c
        distance[i] = distance[i - 1] + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    return distance

def plotBezierCurve(x, y, xCtrl, yCtrl):
    print('xCtrl')
    print(xCtrl)
    print('yCtrl')
    print(yCtrl)
    
    # Create a new figure
    plt.figure()

    # Plot connecting lines
    for i in range(len(xCtrl) - 1):
        plt.plot([xCtrl[i], xCtrl[i+1]], [yCtrl[i], yCtrl[i+1]], color='lightgray', linestyle='--')

    # Plot control points
    plt.plot(xCtrl, yCtrl, 'o', color='lightblue', label='Control Points')
    
    # Plot original points
    plt.plot(x, y, 'bo', label='Original Points')

    # Plot Bézier curve
    tValues = np.linspace(0, 1, 100)
    bezierCurve = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
    fitCurve = np.dot(bezierCurve, np.array([xCtrl, yCtrl]).T)
    plt.plot(fitCurve[:, 0], fitCurve[:, 1], 'g-', label='Bézier Curve')


    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Show the plot
    plt.show()
