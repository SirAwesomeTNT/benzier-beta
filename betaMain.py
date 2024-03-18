import matplotlib.pyplot as plt
import numpy as np
import sys
import bezierCurveMethods as bezierCurveMethods
import ffmpegMethods as ffmpegMethods
import graphicsMethods as graphicsMethods

# Loading functions from the imported modules
# Bezier curve functions
generateControlPoints = bezierCurveMethods.calculateLeastSquaresBezierControlPoints
plotBezierCurve = bezierCurveMethods.plotBezierCurve
# ffmpeg functions
openSongAtFileLocation = ffmpegMethods.openSongAtFileLocation
extractSamples = ffmpegMethods.extractSamples
# graphics functions
plotSamplesAndBezierCurves = graphicsMethods.plotSamplesAndBezierCurves


# ACTUAL CODE
# read the songLocation file and extract songPath
filePath = "bezier-alpha/songLocation.txt"
songPath = openSongAtFileLocation(filePath)

# extract samples from file at songPath
leftChannelSamples, rightChannelSamples = extractSamples(songPath)
left10, right10 = leftChannelSamples[:10], rightChannelSamples[:10]

# plot samples


def generateControlPointsForBezierCurves(samples, step):

    numSamples = len(samples)

    xControlPoints = np.empty((0, 4))
    yControlPoints = np.empty((0, 4))
    indexesIterated = np.empty((0, 4)) # sample x values
    samplesIterated = np.empty((0, 4)) # sample y values

    i = 0
    while i < (numSamples - 3):
        # Get four consecutive samples starting from index i
        indexes = np.arange(i, i+4)
        segment = samples[i:i+4]
        
        # Calculate control points for the Bezier curve segment
        xCtrl, yCtrl = generateControlPoints(indexes.reshape(-1, 1), segment.reshape(-1, 1))

        # Add all values to their respective arrays
        indexesIterated = np.append(indexesIterated, [indexes], axis=0)
        samplesIterated = np.append(samplesIterated, [segment], axis=0)
        xControlPoints = np.append(xControlPoints, [xCtrl], axis=0)
        yControlPoints = np.append(yControlPoints, [yCtrl], axis=0)

        i += step # Iterate by step

    return xControlPoints, yControlPoints, indexesIterated, samplesIterated

for j in range(1, 4):
    xControlPoints, yControlPoints, indexesIterated, samplesIterated = generateControlPointsForBezierCurves(left10, j)
    print(xControlPoints)
    print(yControlPoints)
    print(indexesIterated)
    print(samplesIterated)

    plotSamplesAndBezierCurves(left10, xControlPoints, yControlPoints, indexesIterated, samplesIterated)

# for xCtrl, yCtrl, y, x in zip(xControlPoints, yControlPoints, samplesIterated, indexesIterated):
#     plotBezierCurve(x, y, xCtrl, yCtrl)
# plt.show()

sys.exit()
random_integers = np.random.randint(0, 6, 4)
y = random_integers.reshape(-1, 1)
x = np.arange(4).reshape(-1, 1)

print(x, y)


# run functions
# create control points arrays
xCtrl, yCtrl = generateControlPoints(x, y)

# plot the bezier curve
plotBezierCurve(x, y, xCtrl, yCtrl)