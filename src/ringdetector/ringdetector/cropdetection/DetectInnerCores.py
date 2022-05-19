import cv2
import os

from PIL.Image import Image

from ringdetector.cropdetection.model_config import generate_config
from ringdetector.cropdetection.predictor import CustomizedPredictor
from ringdetector.cropdetection.DetectionProcessor import DetectionProcessor
from ringdetector.Paths import CROP_MODEL

# this is used for the final pipeline:
modelPath = CROP_MODEL
Image.MAX_IMAGE_PIXELS = None

def detectInnerCores(imgPath, csvPath, savePath):
    modelDir = os.path.dirname(modelPath)
    cfg = generate_config(modelDir, (), ())
    cfg.MODEL.WEIGHTS = modelPath

    # initialize custom predictor
    predictor = CustomizedPredictor(cfg)

    img = cv2.imread(imgPath)
    outputs = predictor(img)
    processor = DetectionProcessor(outputs, imgPath, savePath, csvPath)
    processor.filterDetections()
    jsonPath = processor.exportDetections()

    return jsonPath
