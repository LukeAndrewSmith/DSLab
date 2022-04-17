##########################
import torch

TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda # runtime ver

##TODO(2): Distributed Training
#torch.cuda.get_arch_list()  will return the compute capabilities used in your current PyTorch build
#torch.cuda.is_available()
#torch.cuda.device_count()
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

print("torch: ", TORCH_VERSION, 
      "; cuda: ", CUDA_VERSION,
      "; device: ", DEVICE_NAME)

########################
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from model_config import generate_config
from dataset import CropDataset
from ringdetector.cropdetection.operator import CustomizedTrainer#, CustomizedEvaluator

setup_logger()##TODO(2): use wandb logger instead
config = generate_config()##TODO(2): model_config refactoring

##TODO(1): evaluator
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py
# https://github.com/facebookresearch/detectron2/blob/main/tools/plain_train_net.py

data_train, metadata_train = CropDataset(is_train=True).generate_dataset(config.DATASETS.TRAIN)
data_evaluate, metadata_evaluate = CropDataset(is_train=False).generate_dataset(config.DATASETS.TEST)

is_eval_only = False

if is_eval_only:
      model = CustomizedTrainer.build_model(config)
      # config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
      # config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
      ckpt = DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR)
      ckpt.resume_or_load(config.MODEL.WEIGHTS, resume=True)
      
      result = CustomizedTrainer.test(config, model)
      # if config.TEST.AUG.ENABLED:
      #       result.update(CustomizedTrainer.test_with_TTA(config, model))##
      # if comm.is_main_process(): ##TODO(2): Distributed Training 
      #       verify_results(config, result)
      result
else:
      trainer = CustomizedTrainer(config)
      trainer.resume_or_load(resume=False)##
      #     if config.TEST.AUG.ENABLED:
      #         trainer.register_hooks(
      #             [hooks.EvalHook(0, lambda: trainer.test_with_TTA(config, trainer.model))]
      #         )
      trainer.train()
# from detectron2.engine import DefaultPredictor
# predictor = DefaultPredictor(config)##
# evaluator = CustomizedEvaluator(config)##
# evaluator.test(config, predictor.model, evaluators = None)
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# No evaluator found. Use `DefaultTrainer.test(evaluators=)`, or implement its `build_evaluator` method.