from ringdetector.analysis.ImageProcessor import ImageProcessor
from ringdetector.Paths import GENERATED_DATASETS_INNER, DATA
import os


if __name__ == "__main__":
    impath = os.path.join(GENERATED_DATASETS_INNER, "KunA08NN.jpg")
    Im = ImageProcessor(impath)
    Im.denoiseImage()
    Im.computeGradients(method='Canny', threshold1=50, threshold2=100)
    #Im.normalizeGradients()
    Im.plotGradientTimeSeries(Im.gXY)
    Im.saveImage(Im.gXY, os.path.join(DATA, f"{Im.name}_grad_canny.jpg"))
