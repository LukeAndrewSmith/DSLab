##########################
import torch

TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda # runtime ver

## TODO(2): Distributed Training
#torch.cuda.get_arch_list()  will return the compute capabilities used in your current PyTorch build
#torch.cuda.is_available()
#torch.cuda.device_count()
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

print("torch: ", TORCH_VERSION, 
      "; cuda: ", CUDA_VERSION,
      "; device: ", DEVICE_NAME)

########################

import os

from model_config import generate_config
from detectron2.engine import DefaultPredictor

## TODO(2): rotated rect prediction: https://github.com/facebookresearch/detectron2/issues/21, 
## see https://colab.research.google.com/drive/1JXKl48u1fxC35bBryKlQVyQf8tp-DUpE?usp=sharing for a possible solution

config = generate_config()## TODO(2): model_config refactoring

config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

## This is meant for simple demo purposes, see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultPredictor
predictor = DefaultPredictor(config)
## TODO(1): testor
# outputs = predictor(img)