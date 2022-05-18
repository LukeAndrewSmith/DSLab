import torch

def get_cuda_info():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available.")
    else:
        cuda_info = dict()
        cuda_info['torch_version'] = torch.__version__
        cuda_info['cuda_version'] = torch.version.cuda  # runtime ver
        
        device_cnt = torch.cuda.device_count()
        
        if device_cnt == 1:
            cur_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(cur_device)

            cuda_info['device_name'] = device_name
        else:
            ##TODO(2): Distributed Training
            pass
        
        return cuda_info
    
# fix_backend
# random_state = 42
# torch.manual_seed(random_state)
# torch.cuda.manual_seed_all(random_state)
# torch.backends.cudnn.deterministic = True 
# torch.backends.cudnn.benchmark = False