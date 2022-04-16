import torch

TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda # runtime ver

## TODO(2): Distributed Training
#torch.cuda.get_arch_list()  will return the compute capabilities used in your current PyTorch build
#torch.cuda.is_available()
#torch.cuda.device_count()
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

########################
from detectron2.utils.logger import setup_logger
setup_logger()
# TODO(2): use wandb logger instead

from model_config import generate_config
from dataset import CropDataset
from trainer import CustomizedTrainer

print("torch: ", TORCH_VERSION, 
      "; cuda: ", CUDA_VERSION,
      "; device: ", DEVICE_NAME)

config = generate_config()## TODO(2): model_config refactoring

## TODO(1): validation
is_train = True
if is_train:
    data, metadata = CropDataset(name=config.DATASETS.TRAIN, cfg=config, is_train=is_train).generate_dataset()## TODO(1): refactor
    oper = CustomizedTrainer(config)
    oper.resume_or_load(resume=True)
    oper.train()