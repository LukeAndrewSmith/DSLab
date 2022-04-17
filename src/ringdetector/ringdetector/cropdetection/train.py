##########################
import torch

TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda # runtime ver

##TODO(1): better way to handle torch
##TODO(2): Distributed Training
#torch.cuda.get_arch_list()  will return the compute capabilities used in your current PyTorch build
#torch.cuda.is_available()
#torch.cuda.device_count()
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

print("torch: ", TORCH_VERSION, 
      "; cuda: ", CUDA_VERSION,
      "; device: ", DEVICE_NAME)

########################

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

from model_config import generate_config
from dataset import CropDataset
from ringdetector.cropdetection.operator import CustomizedTrainer

setup_logger()##TODO(2): use wandb logger instead
config = generate_config()##TODO(2): model_config refactoring

data_train, metadata_train = CropDataset(is_train=True).generate_dataset(config.DATASETS.TRAIN)
data_evaluate, metadata_evaluate = CropDataset(is_train=False).generate_dataset(config.DATASETS.TEST)
## TODO(2): metadata_train.evaluator_type is None. currently using rotated coco eval as default

## TODO(2): rotated rect prediction: https://github.com/facebookresearch/detectron2/issues/21, 
## see https://colab.research.google.com/drive/1JXKl48u1fxC35bBryKlQVyQf8tp-DUpE?usp=sharing for a possible solution

is_eval_only = True

if is_eval_only:
      model = CustomizedTrainer.build_model(config)
      ckpt = DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR)
      ckpt.resume_or_load(config.MODEL.WEIGHTS, resume=True)
      
      # NOTE: do we need test-time augmentation? diff from build_test_loader?
      # if config.TEST.AUG.ENABLED:
      #       result.update(CustomizedTrainer.test_with_TTA(config, model))##
      # if comm.is_main_process(): ##TODO(2): Distributed Training 
      #       verify_results(config, result)
      
      CustomizedTrainer.test(config, model)
else:
      trainer = CustomizedTrainer(config)
      trainer.resume_or_load(resume=True)
      
      # NOTE: do we need test-time augmentation? diff from build_test_loader?
      #     if config.TEST.AUG.ENABLED:
      #         trainer.register_hooks(
      #             [hooks.EvalHook(0, lambda: trainer.test_with_TTA(config, trainer.model))]
      #         )
      
      trainer.train()