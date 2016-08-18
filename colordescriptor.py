import  numpy as np
import cv2

class ColorDescriptor(object):
    def __init__(self, bins):
        # Store the number of bins for 3D histogram
        self.bins = bins

    def describe(self, image):
        # Covert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int (w * 0.5), int(h * 0.5))

        # Divide the image into four segments
        # (top-left, top-right, bottom-left, bottom-right)
        segments = [(0, cX, 0, cY),
                    (cX, w, 0, cY),
                    (cX, w, cY, h),
                    (0, cX, cY, h)]

        # Construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            # Construct a mask for each corner of image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle = (cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipseMask)


            # Extract a color histogram from the image, then update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # Extract a color histogram from the elliptical region, then update feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return features


    def histogram(self, image, mask):
        # Extract the histogram from HSV color image with its masked region.
        # Using provided bins per channel; then normalize the histogram.

        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()

        return hist
