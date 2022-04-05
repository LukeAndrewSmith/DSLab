from ringdetector.analysis.ImageProcessor import ImageProcessor
from ringdetector.Paths import GENERATED_DATASETS_INNER, DATA
import os
from ringdetector.analysis.EdgeProcessor import EdgeProcessor

from ringdetector.analysis.CoreProcessor import CoreProcessor
from ringdetector.Paths import GENERATED_DATASETS_INNER

TEST_SAMPLE = "KunA01SS"

if __name__ == "__main__":
    cp = CoreProcessor(TEST_SAMPLE)

    cp.scoreCore()
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
