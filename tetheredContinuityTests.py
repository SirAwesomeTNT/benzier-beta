import numpy as np
import ffmpegMethods as ffmpegMethods
import BCMver2 as BCMver2
import matplotlib.pyplot as plt
import forcingTValues as forcingTValues

# Loading functions from the imported modules
# ffmpeg functions
extractAndChunkSamples = ffmpegMethods.extractAndChunkSamples
# faster BCMver2 functions for bezier calculation
fastControlPoints = BCMver2.calculateControlPoints
# forcing t-value functions

class BezierPlot:
    def __init__(self, filePath):
        self.filePath = filePath

    def plotSamplesAndBezierCurves(self, samples, xControlPoints, yControlPoints):
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
        
        ax.set_title('Bezier Curves with Sample Points')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def batchFastControlPoints(self, samples, step):
        """
        Generates control points for Bezier curves based on sample data.

        Args:
        - samples: Array of sample data points
        - step: Integer specifying the step size for sampling

        Returns:
        - xControlPoints: Array of x-coordinate control points
        - yControlPoints: Array of y-coordinate control points
        - indexesIterated: Array of sample indexes used in the control point generation
        - samplesIterated: Array of sample values used in the control point generation
        """

        # Determine the number of samples
        numSamples = len(samples)

        # Initialize arrays to store control points, sample indexes, and sample values
        xControlPoints = np.empty((0, 4))
        yControlPoints = np.empty((0, 4))
        indexesIterated = np.empty((0, 4))  # Sample indexes
        samplesIterated = np.empty((0, 4))  # Sample values

        # Iterate through samples to generate control points
        i = 0
        while i < (numSamples - 3):
            # Get four consecutive samples starting from index i
            indexes = np.arange(i, i+4)
            segment = samples[i:i+4]
            
            # Calculate control points for the Bezier curve segment using the fastControlPoints method
            xCtrl, yCtrl = fastControlPoints(indexes, segment)

            # Append calculated values to their respective arrays
            indexesIterated = np.append(indexesIterated, [indexes], axis=0)
            samplesIterated = np.append(samplesIterated, [segment], axis=0)
            xControlPoints = np.append(xControlPoints, xCtrl, axis=0)
            yControlPoints = np.append(yControlPoints, yCtrl, axis=0)

            i += step  # Move to the next set of samples based on the specified step size

        return xControlPoints, yControlPoints, indexesIterated, samplesIterated

    def matchCurveWithSurroundings(self, xCtrl, yCtrl, curveIndex, beta):
        """
        Adjusts the control points of a Bezier curve to match its surroundings.

        Args:
        - xCtrl: Array of x-coordinate control points of the Bezier curve
        - yCtrl: Array of y-coordinate control points of the Bezier curve
        - curveIndex: Index of the curve in the control points arrays to be modified
        - beta: Coefficient to adjust the control points

        Returns:
        - Updated xCtrl array with modified control points
        - Updated yCtrl array with modified control points
        """

        # Extract relevant curve control points
        xMod, yMod = xCtrl[curveIndex], yCtrl[curveIndex]
        xLeft, yLeft = xCtrl[curveIndex - 1], yCtrl[curveIndex - 1]
        xRight, yRight = xCtrl[curveIndex + 1], yCtrl[curveIndex + 1]

        # Modify left side control point
        xP2, yP2 = xLeft[2], yLeft[2]
        xP3, yP3 = xLeft[3], yLeft[3]

        xMod[1] = xP3 + (xP3 - xP2) * beta
        yMod[1] = yP3 + (yP3 - yP2) * beta

        # Modify right side control point
        xP2, yP2 = xRight[1], yRight[1]
        xP3, yP3 = xRight[0], yRight[0]

        xMod[2] = xP3 + (xP3 - xP2) * beta
        yMod[2] = yP3 + (yP3 - yP2) * beta

        # Update original arrays
        xCtrl[curveIndex] = xMod
        yCtrl[curveIndex] = yMod

        return xCtrl, yCtrl

# declare file path
filePath = "bezier-alpha/songLocation.txt"  
bezierPlot = BezierPlot(filePath)

# extract samples from song at file path
leftChunk, rightChunk = extractAndChunkSamples(bezierPlot.filePath, 13)

# plot samples
xControlPoints, yControlPoints, indexesIterated, samplesIterated = bezierPlot.batchFastControlPoints(leftChunk, 3)
print("xControlPoints:\n", xControlPoints)  # Print x control points for debugging
print("yControlPoints:\n", yControlPoints)  # Print y control points for debugging
print("indexesIterated:\n", indexesIterated)  # Print indexes for debugging
print("samplesIterated:\n", samplesIterated)  # Print samples for debugging

# plot all samples and curves
bezierPlot.plotSamplesAndBezierCurves(leftChunk, xControlPoints, yControlPoints)

# modify array to line up control points with others
xControlPoints, yControlPoints = bezierPlot.matchCurveWithSurroundings(xControlPoints, yControlPoints, 1, 1)
xControlPoints, yControlPoints = bezierPlot.matchCurveWithSurroundings(xControlPoints, yControlPoints, 2, 1)

# plot samples and bezier curves
bezierPlot.plotSamplesAndBezierCurves(leftChunk, xControlPoints, yControlPoints)