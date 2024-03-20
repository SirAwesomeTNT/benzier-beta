import bezierCurveMethods as bezierCurveMethods
import numpy as np
import graphicsMethods as graphicsMethods

# Bezier curve functions
generateControlPoints = bezierCurveMethods.calculateLeastSquaresBezierControlPoints
plotBezierCurve = bezierCurveMethods.plotBezierCurve
# Graphics functions
plotSamplesAndBezierCurves = graphicsMethods.plotSamplesAndBezierCurves

# calculate other control point
def g1Continuity(firstCurve, secondCurve):
        beta = 1
        p2 = firstCurve[2]
        p3 = firstCurve[3]

        secondCurve[1] = p3 + (p3 - p2) * beta

        return secondCurve

# create random samples
samples = np.random.randint(0, 6, 7)

# create samples for first and second bezier
xFirstBezier = np.arange(4)
yFirstBezier = samples[:4]

xSecondBezier = np.arange(3,7)
ySecondBezier = samples[3:7]

# print all arrays
print(samples)
print(xFirstBezier)
print(yFirstBezier)
print(xSecondBezier)
print(ySecondBezier)

# generate control points for first bezier curve
x1Ctrl, y1Ctrl = generateControlPoints(xFirstBezier, yFirstBezier)

# generate control points for second bezier curve
x2Ctrl, y2Ctrl = generateControlPoints(xSecondBezier, ySecondBezier)

# generate control points for third bezier curve (between the other two)
xMiddle = np.arange(2,6)
yMiddle = samples[2:6]
xMCtrl, yMCtrl = generateControlPoints(xMiddle, yMiddle)

# append arrays together
xCtrlAll = np.empty((0, 4))
yCtrlAll = np.empty((0, 4))

xCtrlAll = np.append(xCtrlAll, [x1Ctrl], axis=0)
xCtrlAll = np.append(xCtrlAll, [x2Ctrl], axis=0)
yCtrlAll = np.append(yCtrlAll, [y1Ctrl], axis=0)
yCtrlAll = np.append(yCtrlAll, [y2Ctrl], axis=0)

# middle bezier
# xCtrlAll = np.append(xCtrlAll, [xMCtrl], axis=0)
# yCtrlAll = np.append(yCtrlAll, [yMCtrl], axis=0)

# apply continuity and append
x2Ctrl = g1Continuity(x1Ctrl, x2Ctrl)
y2Ctrl = g1Continuity(y1Ctrl, y2Ctrl)
xCtrlAll = np.append(xCtrlAll, [x2Ctrl], axis=0)
yCtrlAll = np.append(yCtrlAll, [y2Ctrl], axis=0)

print(xCtrlAll)
print(yCtrlAll)

# plot the bezier curves
plotSamplesAndBezierCurves(samples, xCtrlAll, yCtrlAll)