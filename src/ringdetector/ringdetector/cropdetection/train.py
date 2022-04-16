# TODO(2): use wandb logger instead
from detectron2.utils.logger import setup_logger
setup_logger()

from trainer import trainer

trainer.resume_or_load(resume=True)
# trainer.resume_or_load(resume=False)
trainer.train()