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

from detectron2.utils.logger import setup_logger
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from model_config import generate_config
from dataset import CropDataset
from ringdetector.ringdetector.cropdetection.operator import CustomizedTrainer, CustomizedEvaluator

setup_logger()##TODO(2): use wandb logger instead
config = generate_config()##TODO(2): model_config refactoring
is_train = True

if is_train:
      data_train, metadata_train = CropDataset().generate_dataset(is_train=is_train, name=config.DATASETS.TRAIN)
      trainer = CustomizedTrainer(config)
      trainer.resume_or_load(resume=True)
      trainer.train()
else: ##TODO(1): evaluator
      data_evaluate, metadata_evaluate = CropDataset().generate_dataset(is_train=is_train, name=config.DATASETS.TRAIN)
      evaluater = CustomizedEvaluator(config)
      # evaluator.test(cfg, model, evaluators = None)
      # print(inference_on_dataset(predictor.model, val_loader, evaluator))