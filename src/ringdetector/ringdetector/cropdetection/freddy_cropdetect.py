# %%
import numpy as np
import os, json, cv2, random, datetime

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import launch
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import wandb

from ringdetector.cropdetection.D2CustomDataset import D2CustomDataset
from ringdetector.cropdetection.utils import get_cuda_info
from ringdetector.cropdetection.trainer import CustomizedTrainer
from ringdetector.Paths import LABELME_JSONS, POINT_LABELS, D2_RESULTS
from ringdetector.cropdetection.model_config import generate_config

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

setup_logger()
wandb.init('cropdetection', sync_tensorboard=True, settings=wandb.Settings(start_method="thread", console="off"))
data_mode = 'outerInner'
# %%

ds = D2CustomDataset(
    json_path=LABELME_JSONS, pos_path=POINT_LABELS
)
#dataset = ds.generate_dataset("train", "outerInner", False)
DatasetCatalog.register("train", lambda d="train": ds.generate_dataset(d, data_mode, False))
DatasetCatalog.register("val", lambda d="val": ds.generate_dataset(d, data_mode, False))
DatasetCatalog.register("test", lambda d="test": ds.generate_dataset(d, data_mode, False))

metadata_train = MetadataCatalog.get("train")
metadata_train.set(thing_classes=[f"inner_crop"])
metadata_val = MetadataCatalog.get("val")
metadata_val.set(thing_classes=[f"inner_crop"])
metadata_test = MetadataCatalog.get("test")
metadata_test.set(thing_classes=[f"inner_crop"])

"""dataset_train = DatasetCatalog.get("train")
for d in random.sample(dataset_train, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('EdgeGBR', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""


# %%
# create a folder for each run:
res_dir = os.path.join(D2_RESULTS, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(f'results will be stored here: {res_dir}')
cfg = generate_config(res_dir, ("train",), ("val",))
print(get_cuda_info())
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# %%
trainer = CustomizedTrainer(cfg)
trainer.resume_or_load(resume=True)

trainer.train()