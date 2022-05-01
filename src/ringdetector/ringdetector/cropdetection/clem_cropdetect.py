#%%
import numpy as np
import os, json, cv2, random

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.cropdetection.D2CustomDataset import D2CustomDataset
from ringdetector.cropdetection.utils import get_cuda_info
from ringdetector.cropdetection.trainer import CustomizedTrainer
from ringdetector.Paths import LABELME_JSONS, POINT_LABELS, D2_RESULTS


setup_logger()

#%%

ds = D2CustomDataset(
    json_path=LABELME_JSONS, pos_path=POINT_LABELS
)
#dataset = ds.generate_dataset("train", "inner", False)
name = "innercrop_train"
DatasetCatalog.register(name, lambda d="train":ds.generate_dataset(d, "inner", False))
metadata = MetadataCatalog.get(name)
metadata.set(thing_classes=[f"inner_false"])
        

#%%
#for d in random.sample(dataset, 3):
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#    out = visualizer.draw_dataset_dict(d)
#    cv2.imshow('EdgeGBR', out.get_image()[:, :, ::-1])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


# %%
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (name,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

#%%
print(get_cuda_info())

#%%
cfg.OUTPUT_DIR = D2_RESULTS
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomizedTrainer(cfg)
trainer.resume_or_load(resume=True)

trainer.train()