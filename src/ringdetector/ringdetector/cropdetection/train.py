from trainer import CustomizedTrainer
from model_config import cfg

## TODO: model_config refactoring
trainer = CustomizedTrainer(cfg)
trainer.resume_or_load(resume=True)
# trainer.resume_or_load(resume=False)
trainer.train()