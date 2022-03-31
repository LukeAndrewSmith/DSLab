import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from ringdetector.preprocessing.DatasetExtractor import DatasetExtractor
from ringdetector.Paths import GENERATED_DATASETS_INNER, DATA
import scipy.misc
from PIL import Image


def normalize(arr):
    norm = (255 * (arr - np.min(arr)) / np.ptp(arr)).astype(int)
    return norm


if __name__ == "__main__":
    # read in image
    im = cv2.imread(os.path.join(GENERATED_DATASETS_INNER, "KunA08NN.jpg"), cv2.IMREAD_GRAYSCALE)
    print(im.shape)
    #im = cv2.fastNlMeansDenoising(im,None, 10,7,21)
    # get gradient:
    gX = cv2.Sobel(im, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    gX_inv = np.moveaxis(gX, 1,0)

    gY = cv2.Sobel(im, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3)
    gY_inv = np.moveaxis(gY, 1, 0)


    # save as image:
    abs_grad_x = cv2.convertScaleAbs(gX)
    abs_grad_y = cv2.convertScaleAbs(gY)
    grad = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    abs_grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    abs_grad_x = normalize(abs_grad_x)
    abs_grad_y = normalize(abs_grad_y)
    abs_grad = normalize(abs_grad)

    gX = normalize(gX)
    gY = normalize(gY)
    grad = normalize(grad)

    fig, ax = plt.subplots(8, 2)
    for i in range(8):
        ax[i, 0].plot(np.moveaxis(gX, 1,0)[0:1000, i * 20])
        ax[i, 1].plot(np.moveaxis(gY, 1,0)[0:1000, i * 20])
    plt.show()

    abs_grad_xim = Image.fromarray(abs_grad_x.astype(np.uint8))
    abs_grad_yim = Image.fromarray(abs_grad_y.astype(np.uint8))
    abs_gradim = Image.fromarray(abs_grad.astype(np.uint8))

    """abs_grad_xim.save(os.path.join(DATA, "KunA08NN_gradx_abs.jpg"))
    abs_grad_yim.save(os.path.join(DATA, "KunA08NN_grady_abs.jpg"))
    abs_gradim.save(os.path.join(DATA, "KunA08NN_grad_abs.jpg"))"""

    gXim = Image.fromarray(gX.astype(np.uint8))
    gYim = Image.fromarray(gY.astype(np.uint8))
    gradim = Image.fromarray(grad.astype(np.uint8))


    """gXim.save(os.path.join(DATA, "KunA08NN_gradx_nodenoise.jpg"))
    gYim.save(os.path.join(DATA, "KunA08NN_grady_nodenoise.jpg"))
    gradim.save(os.path.join(DATA, "KunA08NN_grad_nodenoise.jpg"))"""


    print(grad)
    cv2.imshow('image', im)
    cv2.waitKey()





