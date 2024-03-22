import numpy as np
import forcingTValues as fTV
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Rational

# declare methods
solveForT = fTV.solveForT
simplifyForControlPoints = fTV.simplifyForControlPoints

def findRelations(start, end, xControlPoints, yControlPoints, indexesIterated, samplesIterated):

    xCoefficients = np.empty((2, 0))
    yCoefficients = np.empty((2, 0))

    for index in range(start, end):
        # method calls
        t1 = solveForT(xControlPoints[index], indexesIterated[index, 1])
        relation1x = simplifyForControlPoints(xControlPoints[index], indexesIterated[index, 1], t1)
        relation1y = simplifyForControlPoints(yControlPoints[index], samplesIterated[index, 1], t1)

        t2 = solveForT(xControlPoints[index], indexesIterated[index, 2])
        relation2x = simplifyForControlPoints(xControlPoints[index], indexesIterated[index, 2], t2)
        relation2y = simplifyForControlPoints(yControlPoints[index], samplesIterated[index, 2], t2)
        # print statements
        print("CurveIndex: ", index)
        print("t1:", t1)
        print("relation1x:", relation1x)
        print("relation1y:", relation1y)
        print("t2:", t2)
        print("relation2x:", relation2x)
        print("relation2y:", relation2y)
        print("-----------------")

    #     currentXCurveArray = np.array([
    #         [indexesIterated[index, 1], indexesIterated[index, 2]],
    #         [relation1x[1], relation2x[1]]
    #     ])
    #     currentYCurveArray = np.array([
    #         [samplesIterated[index, 1], samplesIterated[index, 2]],
    #         [relation1y[1], relation2y[1]]
    #     ])

    #     xCoefficients = np.append(xCoefficients, currentXCurveArray, axis=1)
    #     yCoefficients = np.append(yCoefficients, currentYCurveArray, axis=1)

    # print(xCoefficients)
    # print(yCoefficients)
    return xCoefficients, yCoefficients

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
    curveColors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    # Plot Bezier curves
    for i, (xCtrl, yCtrl) in enumerate(zip(xControlPoints, yControlPoints)):
        curveColor = curveColors[i % len(curveColors)]  # Cycle through the list of colors
        
        # Plot thin grey lines connecting control points
        for j in range(len(xCtrl) - 1):
            ax.plot(xCtrl[j:j+2], yCtrl[j:j+2], color='lightgray', linestyle='--')

        # Plot Bézier curve
        tValues = np.linspace(0, 1, 100)
        bezierCurve = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
        fitCurve = np.dot(bezierCurve, np.array([xCtrl, yCtrl]).T)
        ax.plot(fitCurve[:, 0], fitCurve[:, 1], color=curveColor, label=f'Bezier Curve {i+1}')  # Plot Bézier curve
        
        # Plot control points in the same color as the Bezier curve
        ax.scatter(xCtrl, yCtrl, color=curveColor, label=f'Control Points {i+1}')

    # Plot sample points as black dots
    ax.scatter(np.arange(len(samples)), samples, color='black', label='Samples')
    
    # # Plot lines
    # # P1 Line
    # x_values = np.linspace(-10, 10, 100)  # Adjust the range of x-values as needed
    # y_values = 0.24232 - 0.571737 * x_values
    # ax.plot(x_values, y_values, label='0 = 0.24232 - 0.571737 * x', color='green')

    # # P2 Line
    # x_values = np.linspace(-10, 10, 100)  # Adjust the range of x-values as needed
    # y_values = 0.590842 * x_values - 1.32344
    # ax.plot(x_values, y_values, label='0 = 0.24232 - 0.571737 * x', color='green')

    ax.set_title('Bezier Curves with Sample Points')
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig, ax

xControlPoints = np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
yControlPoints = np.array([[5, Rational(5/6), Rational(-1/3), 6], [6, Rational(35/3), Rational(-31/6), 5]])
indexesIterated = np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
samplesIterated = np.array([[5, 2, 2, 6], [6, 6, 2, 5]])

findRelations(0, 2, xControlPoints, yControlPoints, indexesIterated, samplesIterated)

samplesIterated = np.array([5, 2, 2, 6, 6, 2, 5])

# plot bezier so that it matches samples
# plotSamplesAndBezierCurves(samplesIterated, xControlPoints, yControlPoints)

# change the value of P1 on curve 2 to be C1 continuous with curve P1
xControlPoints[1, 1] = 2 * xControlPoints[0, 3] - xControlPoints[0, 2]
yControlPoints[1, 1] = 2 * yControlPoints[0, 3] - yControlPoints[0, 2]
# plot again
fig, ax = plotSamplesAndBezierCurves(samplesIterated, xControlPoints, yControlPoints)

#

def simplify_equation():
    # Define the symbols
    P1, P2 = sp.symbols('P1 P2')

    # Define the equation
    equation = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * P2 + 0.35**3 * 3 - 1

    # Simplify the equation
    simplified_equation = sp.simplify(equation)
    return simplified_equation

# # Call the method and print the simplified equation
# simplified_equation = simplify_equation()
# print("Simplified equation:", simplified_equation)

def solve_equation():
    P2 = sp.symbols('P2')
    equation = sp.Eq(8.7267 - 0.53846 * P2, 13.296 - 1.8571 * P2)
    P2_solution = sp.solve(equation, P2)
    return P2_solution

# # Call the function to solve for P1
# P2_solution = solve_equation()
# print("Solution for P2:", P2_solution)

# for x
# equation: 0 = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
# assume t = 0.35 at sample point (1, 4) [index 1]
# example:  0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * P2 + 0.35**3 * 3 - 1
# eq. 1     P2 = 3.64783 - 1.85714 * P1
# what does this mean: when t = 0.35, the x values of P1 and P2 are related according to this equation...?

# assume t = 0.65 at sample point (2, 4) [index 2]
# example:  0 = (1 - 0.65)**3 * 0 + 3 * (1 - 0.65)**2 * 0.65 * P1 + 3 * (1 - 0.65) * 0.65**2 * P2 + 0.65**3 * 3 - 2
# eq. 2     P2 = 2.65117 - 0.538462 * P1
# what does this mean: when t = 0.65, the x values of P1 and P2 are related according to this equation...?

# first attempt
# set equal to each other
#           3.64783 - 1.85714 * P1 = 2.65117 - 0.538462 * P1
# equals:   P1x = 0.755802402102712
# what does this mean: no idea

# second attempt
# substitute eq. 1 back into original equation @ t = 0.35
# equation: 0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * P2 + 0.35**3 * 3 - 1
# sub:      0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * (3.64783 - 1.85714 * P1) + 0.35**3 * 3 - 1
# result:   P1x = -313 / 546 or -0.57251908
# is this helpful? not sure...

# for y
# equation: 0 = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
# assume t = 0.35 at sample point (1, 4) [index 1]
# example:  0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * P2 + 0.35**3 * 3 - 4
# eq. 3     P2 = 16.2067 - 1.85714 * P1
# what does this mean: when t = 0.35, the y values of P1 and P2 are related according to this equation...?

# assume t = 0.65 at sample point (2, 4) [index 2]
# example:  0 = (1 - 0.65)**3 * 0 + 3 * (1 - 0.65)**2 * 0.65 * P1 + 3 * (1 - 0.65) * 0.65**2 * P2 + 0.65**3 * 3 - 4
# eq. 4     P2 = 7.15948 - 0.538462 * P1
# what does this mean: when t = 0.65, the y values of P1 and P2 are related according to this equation...?

# first attempt
# set equal to each other
#           16.2067 - 1.85714 * P1 = 7.15948 - 0.538462 * P1
# equals    P1y = 6.86082576641151
# what does this mean: no idea

# second attempt
# substitute eq. 4 back into original equation @ t = 0.35
# equation: 0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * P2 + 0.35**3 * 3 - 1
# sub:      0 = (1 - 0.35)**3 * 0 + 3 * (1 - 0.35)**2 * 0.35 * P1 + 3 * (1 - 0.35) * 0.35**2 * (16.2067 - 1.85714 * P1) + 0.35**3 * 3 - 1
# result:   P1y = -4395605.07351801
# is this helpful? not sure...

# trying first attempts with P2 for both x and y
# for x
# convert eq. 1 to solve for P1
#           P1 = 1.9642 - 0.53846 * P2
# convert eq. 2 to solve for P1
#           P1 = 4.9236 - 1.8571 * P2
# set equations equal to each other
#           1.9642 - 0.53846 * P2 = 4.9236 - 1.8571 * P2
# equals:   P2x = 2.24428198750228

# for y
# convert eq. 3 to solve for P1
#           P1 = 8.7267 - 0.53846 * P2
# convert eq. 4 to solve for P1
#           P1 = 13.296 - 1.8571 * P2
# set equations equal to each other
#           8.7267 - 0.53846 * P2 = 13.296 - 1.8571 * P2
# equals:   P2y = 3.46516107504702

# now the moment of truth...try it in desmos
# aaaaaaaaand it didn't work :((((