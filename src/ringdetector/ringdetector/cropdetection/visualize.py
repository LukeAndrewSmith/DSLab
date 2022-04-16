import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer

## TODO(2): Visualization Method
def visualize(data, metadataset):
    d = data[0]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    plt.figure(figsize=(16,8))
    plt.imshow(out.get_image())
    plt.show()
    plt.close()