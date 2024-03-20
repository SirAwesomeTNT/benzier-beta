import matplotlib.pyplot as plt
import numpy as np
import bezierCurveMethods as bezierCurveMethods
import ffmpegMethods as ffmpegMethods
import graphicsMethods as graphicsMethods
import BCMver2 as BCMver2

# Loading functions from the imported modules
# Bezier curve functions
generateControlPoints = bezierCurveMethods.calculateLeastSquaresBezierControlPoints
plotBezierCurve = bezierCurveMethods.plotBezierCurve
# ffmpeg functions
extractSamples = ffmpegMethods.extractSamples
# graphics functions
plotSamplesAndBezierCurves = graphicsMethods.plotSamplesAndBezierCurves

def generateControlPointsForBezierCurves(samples, step):
    """
    Generates control points based on sample data.

    Args:
    - samples: Array of sample data points
    - step: Integer specifying step size

    Returns:
    - xControlPoints: Array of x-coordinate control points
    - yControlPoints: Array of y-coordinate control points
    - indexesIterated: Array of sample indexes
    - samplesIterated: Array of sample values
    """

    numSamples = len(samples)

    xControlPoints = np.empty((0, 4))
    yControlPoints = np.empty((0, 4))
    indexesIterated = np.empty((0, 4))  # sample x values
    samplesIterated = np.empty((0, 4))  # sample y values

    i = 0
    while i < (numSamples - 3):
        # Get four consecutive samples starting from index i
        indexes = np.arange(i, i+4)
        segment = samples[i:i+4]
        
        # Calculate control points for the Bezier curve segment
        xCtrl, yCtrl = generateControlPoints(indexes, segment)

        # Append all values to their respective arrays
        indexesIterated = np.append(indexesIterated, [indexes], axis=0)
        samplesIterated = np.append(samplesIterated, [segment], axis=0)
        xControlPoints = np.append(xControlPoints, [xCtrl], axis=0)
        yControlPoints = np.append(yControlPoints, [yCtrl], axis=0)

        i += step  # Iterate by step

    return xControlPoints, yControlPoints, indexesIterated, samplesIterated

def samplesFromFilePath(filePath):

    # read and open the song at filePath
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    # extract samples from file at songPath
    leftChannelSamples, rightChannelSamples = extractSamples(songPath)

    return leftChannelSamples, rightChannelSamples

filePath = "bezier-alpha/songLocation.txt"

left, right = samplesFromFilePath(filePath)
left10, right10 = left[:10], right[:10]

# plot samples
xControlPoints, yControlPoints, indexesIterated, samplesIterated = generateControlPointsForBezierCurves(left10, 3)
print(xControlPoints)  # Print x control points for debugging
print(yControlPoints)  # Print y control points for debugging
print(indexesIterated)  # Print indexes for debugging
print(samplesIterated)  # Print samples for debugging

# plot all samples and curves
plotSamplesAndBezierCurves(left10, xControlPoints, yControlPoints)

# modify array to line up control points with others

