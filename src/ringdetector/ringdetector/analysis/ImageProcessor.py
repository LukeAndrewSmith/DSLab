from argparse import ArgumentError
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt


class ImageProcessor:
    """
    class that takes an image and computes all sorts of processing
    """
    def __init__(self, path, cfg):
        self.name = path.split('/')[-1][:-4]
        self.cfg = cfg
        
        if self.cfg.ipread == "grayscale":
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.cfg.ipread == "hsv":
            gbr_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            hsv_img = cv2.cvtColor(gbr_img, cv2.COLOR_BGR2HSV)
            self.image = hsv_img[:,:,2]
        else: 
            self.image = None

        self.denoisedImage = self.__denoiseImage()

        self.gX, self.gY = None, None
        self.gXY = None


    def computeGradients(self):
        if self.cfg.ipgrad == 'Sobel':
            self.gX = self.__computeSobelGradient(direction='x')
            self.gY = self.__computeSobelGradient(direction='y')
            self.gXY = self.__add(self.gX, self.gY)
        elif self.cfg.ipgrad == 'Canny':
            self.gXY = cv2.Canny(
                self.denoisedImage, 
                threshold1=self.cfg.cannymin, 
                threshold2=self.cfg.cannymax
            )

    def __denoiseImage(self):
        #TODO: investigate impact of all three params
        denoisedImage = cv2.fastNlMeansDenoising(
            self.image, None, hVal=self.cfg.denoisehval, 
            templateWindowSize=self.cfg.denoisetempwind, 
            searchWindowSize=self.cfg.denoisesearchwind
        )
        return denoisedImage

    def normalizeGradients(self):
        if self.gX is not None:
            self.gX = self.__normalize(self.gX)
        if self.gY is not None:
            self.gY = self.__normalize(self.gY)
        if self.gXY is not None:
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

    def __computeSobelGradient(self, direction):
        if direction == 'x':
            g = cv2.Sobel(
                self.denoisedImage, ddepth=cv2.CV_16S, dx=1, dy=0, 
                ksize=self.cfg.sobelksize
            )
        elif direction == 'y':
            g = cv2.Sobel(
                self.denoisedImage, ddepth=cv2.CV_16S, dx=0, dy=1,
                ksize=self.cfg.sobelksize
            )
        else:
            print('please provide direction: x or y?')
            return None
        return g


    def __normalize(self, im):
        """
        shifitng by - min and then rescaling with ptp (peak to peak = min max dist)
        """
        imNorm = (255 * (im - np.min(im)) / np.ptp(im)).astype(int)
        return imNorm



