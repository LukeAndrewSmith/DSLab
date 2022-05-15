import argparse
import os
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO)

from ringdetector.preprocessing.InnerCropExtraction import extractInnerCrops
from ringdetector.analysis.InnerCropProcessing import processInnerCrops

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process validation dataset to images showing errors.')
    parser.add_argument('imagePath', type=str,
                        help='Path to the image of cores to process')
    parser.add_argument('savePath', type=str,
                        help='The directory in which to save the results')
    parser.add_argument('--openLabelme', action='store_true',
                        help='Pause and open labelme after auto-detecting cores')
    args = parser.parse_args()

    if not os.path.isfile(args.imagePath):
        print("Error: invalid imagePath")
        exit()
    if not os.path.isdir(args.savePath):
        print("Error: invalid savePath")
        exit()

    logging.info(f"Running with args: imagePath: {args.imagePath}, openLabelme: {args.openLabelme}")
    logging.info(f"Auto-detecting cores")
    
    # TODO: call the auto-crop detection and set the correct labelmeJsonPath
    # labelMeJsonPath = detectCores(args.imagePath)
    labelMeJsonPath = '/Users/lukeasmi/Documents/ETHZ/dslabtreering/data/labelme_jsons/KunA08.json'
    innerCrops = extractInnerCrops(labelmeJsonPath=labelMeJsonPath, openLabelme=args.openLabelme, saveDataset=False)
    logging.info(f"Identifying rings")
    processInnerCrops(innerCrops, args.savePath)
