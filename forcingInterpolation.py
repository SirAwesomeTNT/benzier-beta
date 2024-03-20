import numpy as np
import bezierCurveMethods as bCM
plotBezierCurve = bCM.plotBezierCurve

x_samples = np.zeros((4,1), dtype=np.float32)
y_samples = np.zeros((4,1), dtype=np.float32)
x_control = np.zeros((4,1), dtype=np.float32)
y_control = np.zeros((4,1), dtype=np.float32)

x_samples = np.array([[0],[1.5],[5],[9]])
y_samples = np.array([[1.5],[2],[0],[-0.5]])

coefficents = np.array([[6, 0, 0, 0], [-5, 18, -9, 2], [2, -9, 18, -5], [0, 0, 0, 6]])

x_control = coefficents @ x_samples / 6
y_control = coefficents @ y_samples / 6

plotBezierCurve(x_samples, y_samples, x_control, y_control)