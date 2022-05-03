from random import sample
import cv2
import matplotlib.pyplot as plt
import json, os
from collections import defaultdict
from detectron2.utils.visualizer import Visualizer
import matplotlib.patches as patches
from ringdetector.Paths import IMAGES, D2_RESULTS
from pycocotools.coco import COCO




def json_parser(result_path):
    with open(result_path, 'r') as f:
        data = json.load(f)
        grouped = defaultdict(list)
        for item in data:
            grouped[item["image_id"]].append({
                "bbox": item["bbox"],
                "score": item["score"]
            })

    return grouped


def visualize_json(result_path, img_path, k=1):
    data = json_parser(result_path)
    data_samples = sample(data.items(), k)

    for img_name, bboxes in data_samples:
        img = cv2.imread(os.path.join(img_path, img_name))[:, :, ::-1]

        fig, ax = plt.subplots(figsize=(24, 8))
        ax.imshow(img)

        for item in bboxes:
            x0, y0, width, height = item["bbox"]  ##NOTE: XYWH_ABS

            rect = patches.Rectangle((x0 + height, y0), width, height, linewidth=1, edgecolor='yellow',
                                     facecolor='none')
            ax.add_patch(rect)

            ax.text(x0 + width // 2, y0 + height // 2, "score: {score}".format(score=item["score"]),
                    horizontalalignment='center', verticalalignment='center', fontsize=10)

        plt.show()

    plt.close()


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
    #outputs = predictor(data_loader)
    #print(outputs)
    for d in data_samples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadataset, scale=0.5)
        
        outputs = predictor(img)
        result = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(16,8))
        plt.imshow(result.get_image())
        plt.show()
    
    plt.close()


def coco_vis(result_path, img_dir):
    coco = COCO(result_path)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories for damages: \n{}\n'.format(', '.join(nms)))


if __name__ == "__main__":
    res_dir = os.path.join(D2_RESULTS, "2022-05-02_18-45-19")
    visualize_json(
        result_path=os.path.join(res_dir, 'coco_instances_results.json'),
        img_path=IMAGES,
        k=3
    )
    #coco_vis(result_path=os.path.join(res_dir, 'coco_instances_results.json'), img_dir=IMAGES)
