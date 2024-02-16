import matplotlib.pyplot as plt
import numpy as np
import runpy

random_integers = np.random.randint(0, 6, 4)
y = random_integers.reshape(-1, 1)
x = np.arange(4).reshape(-1, 1)

# load file
bezierCurveMethods = runpy.run_path("bezier-beta/bezierCurveMethods.py")

# call function
generateControlPoints = bezierCurveMethods['calculateLeastSquaresBezierControlPoints']
plotBezierCurve = bezierCurveMethods['plotBezierCurve']

# run functions
# create control points array
control = generateControlPoints(x, y)
# split the control points array into two arrays, with the top half containing x-values and the bottom half containing y-values
xCtrl, yCtrl = control[:, 0], control[:, 1]

plotBezierCurve(x, y, xCtrl, yCtrl)

# open file for audio track

# extract and divide samples

# select a random selection of samples

# approximate and display curve