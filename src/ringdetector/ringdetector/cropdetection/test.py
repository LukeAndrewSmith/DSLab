import os

from model_config import generate_config
from detectron2.engine import DefaultPredictor

config = generate_config()## TODO(2): model_config refactoring

config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

## This is meant for simple demo purposes, see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultPredictor
predictor = DefaultPredictor(config)
## TODO(1): testor
# outputs = predictor(img)