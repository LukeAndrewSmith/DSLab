import argparse
import os
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO)

from ringdetector.Paths import CORE_LISTS
from ringdetector.cropdetection.DetectInnerCores import detectInnerCores
from ringdetector.preprocessing.InnerCropExtraction import extractInnerCrops
from ringdetector.analysis.InnerCropProcessing import inferInnerCrops

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process validation dataset to images showing errors.')
    parser.add_argument('-imagePath', type=str,
                        help='Path to the image of cores to process')
    parser.add_argument('-savePath', type=str,
                        help='The directory in which to save the results')
    parser.add_argument('--openLabelme', action='store_true',
                        help='Pause and open labelme after auto-detecting cores')
    args = parser.parse_args()

    if not os.path.isfile(args.imagePath):
        logging.error("Error: invalid imagePath")
        exit()
    
    imageName = os.path.basename(args.imagePath)[:-4]
    csvPath = os.path.join(CORE_LISTS, imageName + ".csv")
    if not os.path.isfile(csvPath):
        logging.error("Error: no matching CSV")
        exit()
    if not os.path.isdir(args.savePath):
        logging.error("Error: invalid savePath")
        exit()

    logging.info(f"Running with args: imagePath: {args.imagePath}, "
        f"openLabelme: {args.openLabelme}")
    logging.info(f"Auto-detecting cores")
    
    labelMeJsonPath = detectInnerCores(args.imagePath, csvPath)

    if labelMeJsonPath and args.openLabelme:
        os.system("echo Opening labelme. Please be patient for one moment, labelme can be slow to start")
        os.system(f'labelme {labelMeJsonPath} --logger-level fatal &') # Open in background TODO: maybe detect if windows and change the command... also not sure if this will work in docker?
        input("Press Enter to continue...")

    innerCrops = extractInnerCrops(
        labelmeJsonPath=labelMeJsonPath, 
        csvPath=csvPath,
        saveDataset=False
    )
    logging.info(f"Identifying rings")
    inferInnerCrops(innerCrops, args.savePath)
    logging.info(f"Finished.")
