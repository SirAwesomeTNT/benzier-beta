from sympy import symbols, solve, simplify, Rational

def solveForT(controlPoints, xValue):
    # Define symbols
    t, x, P0, P1, P2, P3 = symbols('t x P0 P1 P2 P3')
    
    # Define the equation
    equation = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
    print("Original Equation:")
    print(equation)

    # Substitute known values into the equation
    equationWithKnowns = equation.subs({P0: controlPoints[0], P1: controlPoints[1], P2: controlPoints[2], P3: controlPoints[3], x: xValue})
    print("\nEquation with Known Values Substituted:")
    print(equationWithKnowns)

    # Solve the equation for t
    solutions = solve(equationWithKnowns, t)
    print("\nSolutions:")
    print(solutions)
    print("-----------------")
    
    return solutions

def simplifyForControlPoints(controlPoints, xValue, tValue):
    # Define symbols
    t, x, P0, P1, P2, P3 = symbols('t x P0 P1 P2 P3')

    for tSolution in tValue:
        print(f"t = {tSolution}")
        # Define the equation with known constants
        equation = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
        print("Original Equation:")
        print(equation)

        print({P0: controlPoints[0], P3: controlPoints[3], x: xValue, t: tSolution})
        # Substitute known values into the equation
        equationWithKnowns = equation.subs({P0: controlPoints[0], P3: controlPoints[3], x: xValue, t: tSolution})
        print("\nEquation with Known Values Substituted:")
        print(equationWithKnowns)

        # Simplify the equation
        simplifiedEquation = simplify(equationWithKnowns)
        print("\nSimplified Equation:")
        print(simplifiedEquation)
        print("-----------------")

    return simplifiedEquation

# based on sample data from audio file:
# xControlPoints:  [[0. 1. 2. 3.]]
# yControlPoints:  [[ 0.         -0.00017293  0.000295   -0.0005188 ]]
# indexesIterated: [[0. 1. 2. 3.]]
# samplesIterated: [[ 0.00000000e+00 -3.05175781e-05 -6.10351562e-05 -5.18798828e-04]]
# using intersection of curve at first point (1, -3.05175781e-05) as example

# # setup for determining x-aspect of curve
# # Known values for every x-component of the curve
# controlPoints = [0, 1, 2, 3]
# xValue = 1
# # solve for t value at point (1, -3.05175781e-05)
# solveForT(controlPoints, xValue) # result: t = 1/3 (only)
# tValue = Rational(1, 3)
# # use t value to solve for control point relation
# simplifyForControlPoints(controlPoints, xValue, tValue)# result: 4*P1/9 + 2*P2/9 - 8/9 = 0

# # setup for determining y-aspect of curve
# # Known values for every y-component of the curve
# controlPoints = [0, -0.00017293, 0.000295, -0.0005188]
# xValue = -3.05175781e-05
# # solve for t value at point (1, -3.05175781e-05)
# solveForT(controlPoints, xValue) # result: t = 1/3 (and 0.08 and 0.59)
# tValue = Rational(1, 3)
# # use t value to solve for control point relation
# simplifyForControlPoints(controlPoints, xValue, tValue)# result: 4*P1/9 + 2*P2/9 + 1.13e-5 = 0