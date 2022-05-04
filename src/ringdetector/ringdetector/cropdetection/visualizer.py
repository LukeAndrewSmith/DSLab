import json, os, cv2
from random import sample
import matplotlib.pyplot as plt
from collections import defaultdict
from detectron2.utils.visualizer import Visualizer
import matplotlib.patches as patches


def jsonParser(resultPath):
    with open(resultPath, 'r') as f:
        data = json.load(f)
        grouped = defaultdict(list)
        for item in data:
            grouped[item["image_id"]].append({
                "bbox": item["bbox"],
                "score": item["score"]
            })

    return grouped


def visualizeJson(resultPath, imgPath, k=1):
    # for now this only works for the bboxes not the segmentation
    data = jsonParser(resultPath)
    dataSamples = sample(data.items(), k)

    for imgName, bboxes in dataSamples:
        img = cv2.imread(os.path.join(imgPath, imgName))[:, :, ::-1]

        fig, ax = plt.subplots(figsize=(24, 8))
        ax.imshow(img)

        for item in bboxes:
            x0, y0, width, height = item["bbox"]  ## NOTE: XYWH_ABS

            rect = patches.Rectangle((x0 + height, y0), width, height, linewidth=1, edgecolor='yellow',
                                     facecolor='none')
            ax.add_patch(rect)

            ax.text(x0 + width // 2, y0 + height // 2, "score: {score}".format(score=item["score"]),
                    horizontalalignment='center', verticalalignment='center', fontsize=10)

        plt.show()

    plt.close()


def visualizeAnno(data, metadataset, k=1):
    dataSamples = sample(data, k)
    
    for d in dataSamples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
        
        result = visualizer.draw_dataset_dict(d)

        plt.figure(figsize=(16,8))
        plt.imshow(result.get_image())
        plt.show()
    
    plt.close()


def visualizePred(data, metadataset, predictor, k=1):
    dataSamples = sample(data, k)
    for d in dataSamples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
        
        outputs = predictor(img)
        result = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(16,8))
        plt.imshow(result.get_image())
        plt.show()
    
    plt.close()


def wandbVisualizePred(data, metadataset, predictor, k=1):
    # same as above but return the im don't plot it so it can be logged to wandb
    # can do sthg like this if wished as well: https://docs.wandb.ai/guides/track/log/media
    dataSamples = sample(data, k)
    results = []
    for d in dataSamples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)

        outputs = predictor(img)
        result = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        results.append(result.get_image())

    return results

