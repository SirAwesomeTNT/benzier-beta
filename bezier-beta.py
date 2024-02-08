import matplotlib.pyplot as plt
import numpy as np
import runpy

random_integers = np.random.randint(0, 6, 4)
y = random_integers.reshape(-1, 1)
x = np.arange(4).reshape(-1, 1)

# execute file
bezierCurveMethods = runpy.run_path("bezier-beta/bezierCurveMethods.py")

# call function
generateControlPoints = bezierCurveMethods['calculateLeastSquaresBezierControlPoints']

# run function
xControl, yControl = generateControlPoints(x, y)

print(xControl, yControl)

# open file for audio track

# extract and divide samples

# select a random selection of samples

# approximate and display curve