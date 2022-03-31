import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt


class ImageProcessor:
    """
    class that takes an image and computes all sorts of processing
    """
    def __init__(self, path):
        self.name = path.split('/')[-1][:-4]
        # reads grayscale for now
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.gX, self.gY = None, None
        self.gXY = None

    def computeGradients(self, ksize=3, method='Sobel'):
        self.gX = self.__computeDirectionalGradient(direction='x', ksize=ksize, method=method)
        self.gY = self.__computeDirectionalGradient(direction='y', ksize=ksize, method=method)
        self.gXY = self.__add(self.gX, self.gY)

    def denoiseImage(self, hVal=10, templateWindowSize=7, searchWindowSize=21):
        self.image = cv2.fastNlMeansDenoising(self.image, None, hVal, templateWindowSize, searchWindowSize)

    def normalizeGradients(self):
        self.gX = self.__normalize(self.gX)
        self.gY = self.__normalize(self.gY)
        self.gXY = self.__normalize(self.gXY)

    def absValGradients(self):
        self.gX = self.__absVal(self.gX)
        self.gY = self.__absVal(self.gY)
        self.gXY = self.__absVal(self.gXY)

    def plotGradientTimeSeries(self, gradient):
        # plots some samples as a time series
        fig, ax = plt.subplots(8, 1)
        height = np.shape(gradient)[0]
        mod = int(np.floor(height / 8.0))
        for i in range(8):
            ax[i].plot(np.moveaxis(gradient, 1, 0)[0:1000, i * mod])
        plt.show()

    def saveImage(self, im, path):
        gradImage = Image.fromarray(im.astype(np.uint8))
        gradImage.save(path)

    def __add(self, im1, im2):
        imSum = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)
        return imSum

    def __absVal(self, im):
        imAbs = cv2.convertScaleAbs(im)
        return imAbs

    def __computeDirectionalGradient(self, direction, ksize=3, method='Sobel'):
        if method == 'Sobel':
            if direction == 'x':
                g = cv2.Sobel(self.image, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=ksize)
            elif direction == 'y':
                g = cv2.Sobel(self.image, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=ksize)
            else:
                print('please provide direction: x or y?')
                return None
            return g
        else:
            print('method not supported yet')
            return None

    def __normalize(self, im):
        """
        shifitng by - min and then rescaling with ptp (peak to peak = min max dist)
        """
        imNorm = (255 * (im - np.min(im)) / np.ptp(im)).astype(int)
        return imNorm



