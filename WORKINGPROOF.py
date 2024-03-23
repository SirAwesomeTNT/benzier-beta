import numpy as np
import sympy as sp
import forcingTValues as fTV
import matplotlib.pyplot as plt
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

def plotBezierCurvesAndSamples(ax, xControlPoints, yControlPoints, samples):
    """
    Plot Bezier curves along with control points and sample points.
    
    Args:
    - ax: Axis object to plot on
    - xControlPoints: List of arrays containing x-coordinate control points for each Bezier curve
    - yControlPoints: List of arrays containing y-coordinate control points for each Bezier curve
    - samples: Array of sample data points
    """
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

    # Update the legend
    ax.legend()

    # Add gridlines
    ax.grid(True)

def plotParametric(ax, equationX, equationY, startT, endT, xTranslation=0, yTranslation=0, equation_color='r', equation_label=None):
    """
    Plot lines for provided equations.
    
    Args:
    - ax: Axis object to plot on
    - equationX: equation for x-values of parametric
    - equationY: equation for y-values of parametric
    - startT: Start value of parameter t
    - endT: End value of parameter t
    - equation_color: Color of the line for the equation
    - equation_label: Label for each equation
    """
    # Convert SymPy equations to lambda functions
    t_symbol = sp.symbols('t')
    equationX_lambda = sp.lambdify(t_symbol, equationX, 'numpy')
    equationY_lambda = sp.lambdify(t_symbol, equationY, 'numpy')

    # Generate values for parameter t
    t_values = np.linspace(startT, endT, 100)

    # Calculate x and y coordinates using the parametric equations
    # x_values = equationX_lambda(t_values) + xTranslation
    # y_values = equationY_lambda(t_values) + yTranslation
    x_values = equationX_lambda(t_values)
    y_values = equationY_lambda(t_values)

    # Plot the parametric curve
    ax.plot(x_values, y_values, color=equation_color, label=equation_label)
    
    # Plot start and end points
    # Starts at light grey, ends at dark grey
    ax.scatter(x_values[0], y_values[0], color='darkgrey', label='start t')
    ax.scatter(x_values[len(x_values)-1], y_values[len(y_values)-1], color='#666666', label='end t')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Parametric Plot')

    # Show plot
    plt.grid(True)
    # plt.axis('equal')  # Equal aspect ratio
    plt.legend()

# declare plot
fig, ax = plt.subplots(figsize=(8, 6))

# declare data
xControlPoints = np.array([[0, 1, 2, 3], [3, 4, 5, 6]]).astype(np.float32)
yControlPoints = np.array([[5, Rational(5/6), Rational(-1/3), 6], [6, Rational(35/3), Rational(-31/6), 5]])
print(type(yControlPoints))
indexesIterated = np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
samplesIterated = np.array([[5, 2, 2, 6], [6, 6, 2, 5]])

# change the shape of samples so that it works with plotSamplesAndBezierCurves method
indexesIterated = np.array([0, 1, 2, 3, 4, 5, 6])
samplesIterated = np.array([5, 2, 2, 6, 6, 2, 5])

# change the value of P1 on curve 2 to be C1 continuous with curve P1
xControlPoints[1, 1] = 2 * xControlPoints[0, 3] - xControlPoints[0, 2]
yControlPoints[1, 1] = 2 * yControlPoints[0, 3] - yControlPoints[0, 2]

# plot bezier curves
# plotBezierCurvesAndSamples(ax, xControlPoints, yControlPoints, samplesIterated)

# plot the line for all possible values of p1 that go through 'a' (sample @ index 4)
# equation: 0 = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
# equation: P2 = (- (1 - t)**3 * P0 - 3 * (1 - t)**2 * t * P1 - t**3 * P3 + x) / (3 * (1 - t) * t**2)
# sub:      P2 = (- (1 - t)**3 * xControlPoints[1, 0] - 3 * (1 - t)**2 * t * xControlPoints[1, 1] - t**3 * xControlPoints[1, 3] + indexesIterated[4]) / (3 * (1 - t) * t**2)

def simplify_equation(equation):
    # Extract symbols from the equation
    symbols = list(equation.free_symbols)

    # Simplify the equation
    simplified_equation = sp.simplify(equation)
    return simplified_equation

def solve_equation(equation, variables):
    """
    Solve an equation for a given variable.

    Args:
    - equation (sympy.Expr): The equation to solve.
    - variable (sympy.Symbol): The variable to solve for.

    Returns:
    - solutions (list): List of solutions for the variable.
    """
    solutions = sp.solve(equation, variables, domain=sp.Interval(0, 1) * sp.Interval(0, 1))
    print("Solutions for:", equation)
    for sol in solutions:
        print(sol)

    return solutions

# Define the equation
t, P2 = sp.symbols('t P2')
# equation of P2x that goes through a
equationAx = sp.Eq(P2, (- (1 - t)**3 * xControlPoints[1, 0] - 3 * (1 - t)**2 * t * xControlPoints[1, 1] - t**3 * xControlPoints[1, 3] + indexesIterated[4]) / (3 * (1 - t) * t**2))
print("equationAx:", simplify_equation(equationAx))
# equation of P2y that goes through a
equationAy = sp.Eq(P2, (- (1 - t)**3 * yControlPoints[1, 0] - 3 * (1 - t)**2 * t * yControlPoints[1, 1] - t**3 * yControlPoints[1, 3] + samplesIterated[4]) / (3 * (1 - t) * t**2))
print("equationAy:", simplify_equation(equationAy))
# equation of P2x that goes through b
equationBx = sp.Eq(P2, (- (1 - t)**3 * xControlPoints[1, 0] - 3 * (1 - t)**2 * t * xControlPoints[1, 1] - t**3 * xControlPoints[1, 3] + indexesIterated[5]) / (3 * (1 - t) * t**2))
print("equationBx:", simplify_equation(equationBx))
# equation of P2y that goes through b
equationBy = sp.Eq(P2, (- (1 - t)**3 * yControlPoints[1, 0] - 3 * (1 - t)**2 * t * yControlPoints[1, 1] - t**3 * yControlPoints[1, 3] + samplesIterated[5]) / (3 * (1 - t) * t**2))
print("equationBy:", simplify_equation(equationBy))

# set equations equal to each other and simplify
# All P2 that goes through a = All P2 that goes through b
# variables: t is now ta, tb for each respective point
ta, tb = sp.symbols('ta tb')
# intersection should be solution that goes through both
# equationP2x = sp.Eq((- (1 - ta)**3 * xControlPoints[1, 0] - 3 * (1 - ta)**2 * ta * xControlPoints[1, 1] - ta**3 * xControlPoints[1, 3] + indexesIterated[4]) / (3 * (1 - ta) * ta**2), (- (1 - tb)**3 * xControlPoints[1, 0] - 3 * (1 - tb)**2 * tb * xControlPoints[1, 1] - tb**3 * xControlPoints[1, 3] + indexesIterated[5]) / (3 * (1 - tb) * tb**2))
# simpleEqutionP2x = simplify_equation(equationP2x)
# print("equationP2x:", simpleEqutionP2x)

t_symbol = sp.symbols('t')

# instead, do a quick grid search for initial intersections
# for x
# define t-values
taValues = np.linspace(0.01, 0.99, 500)
tbValues = np.linspace(0.01, 0.99, 500)

# Define lambda functions
lambdaAx = sp.lambdify(t_symbol, equationAx.rhs, modules=['numpy'])
lambdaBx = sp.lambdify(t_symbol, equationBx.rhs, modules=['numpy'])
lambdaAy = sp.lambdify(t_symbol, equationAy.rhs, modules=['numpy'])
lambdaBy = sp.lambdify(t_symbol, equationBy.rhs, modules=['numpy'])

p1x = xControlPoints[1,1]
p1y = yControlPoints[1,1].evalf()
print(p1x, p1y)

# define solution arrays
taSolutions = np.array([])
tbSolutions = np.array([])
distanceFromP1 = np.array([])

# find solutions
for taVal in taValues:
    aX = lambdaAx(taVal)
    aY = lambdaAy(taVal)
    for tbVal in tbValues:
        bX = lambdaBx(tbVal)
        bY = lambdaBy(tbVal)
        if (bX - aX)**2 + (bY - aY)**2 < 5e-4:
            print(f"P2a: ({aX}, {aY})")
            print(f"P2b: ({bX}, {bY})")
            distanceSquared = (p1x - aX)**2 + (p1y - aY)**2
            taSolutions = np.append(taSolutions, taVal)
            tbSolutions = np.append(tbSolutions, tbVal)
            distanceFromP1 = np.append(distanceFromP1, distanceSquared)

# for all these t-values, the resulting aX = bX and aY = bY
print("taSolutions:", taSolutions)
print("tbSolutions:", tbSolutions)
print("distanceFromP1:", distanceFromP1)

# select closest solution
minIndex = np.argmin(distanceFromP1)
taFinal = taSolutions[minIndex]
tbFinal = tbSolutions[minIndex]
print("taFinal:", taFinal)
print("tbFinal:", tbFinal)

# solve again for x-values
print(xControlPoints)
xControlPoints[1, 2] = lambdaAx(taSolutions[minIndex])
yControlPoints[1, 2] = lambdaAy(taSolutions[minIndex])
print(xControlPoints)
plotBezierCurvesAndSamples(ax, xControlPoints, yControlPoints, samplesIterated)

# the solution we want is at: (4.6, -4.5)


# for later:
# conduct another test to find a better/closer solution
# plug exact ta/tb into equation once more to find final P2
# implement implement implement!



# PROOF INTERSECTION!!!
# plot parametric for a
plotParametric(ax, equationAx.rhs, equationAy.rhs, 0.3, 0.8, indexesIterated[4], samplesIterated[4])
# plot parametric for b
plotParametric(ax, equationBx.rhs, equationBy.rhs, 0.45, 0.85, indexesIterated[5], samplesIterated[5], equation_color='purple')
plt.show()

# whole parametric curves
# # plot parametric for a
# plotParametric(ax, equationAx.rhs, equationAy.rhs, 0.1, 0.90, indexesIterated[4], samplesIterated[4])
# # plot parametric for b
# plotParametric(ax, equationBx.rhs, equationBy.rhs, 0.2, 0.90, indexesIterated[5], samplesIterated[5], equation_color='purple')
# plt.show()

####### notes belowwwwww

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