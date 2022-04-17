import torch

##TODO(2): Distributed Training
#torch.cuda.is_available()
#torch.cuda.device_count()
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

print("torch: ", torch.__version__, 
      "; cuda: ", torch.version.cuda, # runtime ver
      "; device: ", DEVICE_NAME)