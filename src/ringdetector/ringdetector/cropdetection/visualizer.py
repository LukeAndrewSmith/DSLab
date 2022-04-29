from random import sample

import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer

##TODO(2): refactoring
def visualize_anno(data, metadataset, k=1):
    data_samples = sample(data, k)
    
    for d in data_samples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
        
        result = visualizer.draw_dataset_dict(d)

        plt.figure(figsize=(16,8))
        plt.imshow(result.get_image())
        plt.show()
    
    plt.close()

def visualize_pred(data, metadataset, predictor, k=1):
    data_samples = sample(data, k)
    
    for d in data_samples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
        
        outputs = predictor(img)
        result = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(16,8))
        plt.imshow(result.get_image())
        plt.show()
    
    plt.close()