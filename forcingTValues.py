from sympy import symbols, solve, simplify, Rational

def solveForT(P0Value, P1Value, P2Value, P3Value, xValue):
    # Define symbols
    t, x, P0, P1, P2, P3 = symbols('t x P0 P1 P2 P3')
    
    # Define the equation
    equation = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
    print("Original Equation:")
    print(equation)

    # Substitute known values into the equation
    equationWithKnowns = equation.subs({P0: P0Value, P1: P1Value, P2: P2Value, P3: P3Value, x: xValue})
    print("\nEquation with Known Values Substituted:")
    print(equationWithKnowns)

    # Solve the equation for t
    solutions = solve(equationWithKnowns, t)
    print("\nSolutions:")
    print(solutions)
    
    return solutions

def simplifyForControlPoints(P0Value, P3Value, xValue, tValue):
    # Define symbols
    t, x, P0, P1, P2, P3 = symbols('t x P0 P1 P2 P3')

    # Define the equation with known constants
    equation = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3 - x
    print("Original Equation:")
    print(equation)

    # Substitute known values into the equation
    equationWithKnowns = equation.subs({P0: P0Value, P3: P3Value, x: xValue, t: tValue})
    print("\nEquation with Known Values Substituted:")
    print(equationWithKnowns)

    # Simplify the equation
    simplifiedEquation = simplify(equationWithKnowns)
    print("\nSimplified Equation:")
    print(simplifiedEquation)

    return simplifiedEquation

# Known values for solving for control points
p0Value = Rational(0, 1)
p3Value = Rational(3, 1)
xValue = Rational(1, 1)
tValue = Rational(1, 3)
# Equation: 4*P1/9 + 2*P2/9 - 8/9 = 0
simplifyForControlPoints(p0Value, p3Value, xValue, tValue)

print("-----------------")

# Known values for solving for t
p0Value = 0
p1Value = -0.00017293
p2Value = 0.000295
p3Value = -0.0005188
xValue = -3.05175781e-05
# t = 1/3

solveForT(p0Value, p1Value, p2Value, p3Value, xValue)
