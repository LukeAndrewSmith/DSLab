import os
import argparse
import warnings
import logging
import coloredlogs


from ringdetector.Paths import GENERATED_DATASETS_INNER
from ringdetector.analysis.CoreProcessor import CoreProcessor

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



TEST_SAMPLE = "RUEP03WW"

parser = argparse.ArgumentParser()

def getArgs(parser):
    
    ###########################################################################
    # Dataset
    ########################################################################### 
    parser.add_argument('-dataset', type=str, choices=['inner'],
        default='inner', help="Which dataset to load.")
    
    ###########################################################################
    # Image Processor
    ########################################################################### 
    #TODO: experiment with args, pick best default
    parser.add_argument("-ipread", type=str, choices=['grayscale', 'hsv'],
        default='hsv', help="ImageProcessor: grayscale or hsv value component before processing")
    parser.add_argument("-denoisehval", type=int, default=10, 
        help="ImageProcessor: hVal arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisetempwind", type=int, default=7, 
        help="ImageProcessor: templateWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisesearchwind", type=int, default=21, 
        help="ImageProcessor: searchWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-ipgrad", type=str, default="Canny", 
        choices=["Canny", "Sobel"], 
        help="ImageProcessor: gradient-based edge detection method")
    parser.add_argument("-sobelksize", type=int, default=3, 
        help="Ksize arg for cv2.Sobel")
    parser.add_argument("-cannymin", type=int, default=50, 
        help="Minimum threshold (threshold1 arg) for cv2.Canny")
    parser.add_argument("-cannymax", type=int, default=100, 
        help="Maximum threshold (threshold2 arg) for cv2.Canny")

    ###########################################################################
    # Edge Processor
    ########################################################################### 
    parser.add_argument("-minedgelen", type=int, default=80, 
        help="Minimum threshold (threshold1 arg) for cv2.Canny")


cfg = get_args(parser)


if __name__ == "__main__":
    cp = CoreProcessor(TEST_SAMPLE)

    cp.scoreCore()
    print(f"Sample {TEST_SAMPLE}: prec {cp.precision}, rec {cp.recall}")
    cp.exportCoreImg(os.path.join(GENERATED_DATASETS_INNER, "results"))

    # Freddy's old pipeline
    #impath = os.path.join(GENERATED_DATASETS_INNER, "KunA08NN.jpg")
    #Im = ImageProcessor(impath)
    #Im.denoiseImage()
    #Im.computeGradients(method='Canny', threshold1=50, threshold2=100)
    #Im.normalizeGradients()
    #Im.plotGradientTimeSeries(Im.gXY)
    #Im.saveImage(Im.gXY, os.path.join(DATA, f"{Im.name}_grad_canny.jpg"))
    #edgeProcessor = EdgeProcessor(Im.gXY)
    #edgeProcessor.processEdgeInstances(minLength=50)
    #instancepath = os.path.join(DATA, f"{Im.name}_grad_canny_instances_filtered_50.jpg")
    #edgeProcessor.saveEdgeInstanceImage(instancepath)
