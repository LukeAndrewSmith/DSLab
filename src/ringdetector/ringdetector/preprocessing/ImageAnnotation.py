import json
import os
import logging
from copy import deepcopy

from ringdetector.preprocessing.CoreAnnotation import CoreAnnotation
from ringdetector.utils.csvLoader import loadImageCSV

from ringdetector.Paths import POINT_LABELS

# this is a class that reads in a json annotation of an image and creates a class for each core with all relevant
# annotations

class ImageAnnotation:
    def __init__(self, json_path, csv_path=None):

        csv_core_names, csv_first_years = [], []
        if csv_path: 
            csv_core_names, csv_first_years = loadImageCSV(csv_path)

        with open(json_path) as f:
            labelMeAnnotations = json.load(f)
        
        self.image_path = self.__get_image_path(labelMeAnnotations)
        self.image_name = os.path.basename(self.image_path)
        self.height = self.__get_image_height(labelMeAnnotations)
        self.width = self.__get_image_width(labelMeAnnotations)

        labelme_core_names = self.__get_labelme_core_names(labelMeAnnotations)
        aligned_cores = self.__align_core_info(
            labelme_core_names, csv_core_names, csv_first_years
        )
        
        self.core_annotations = self.__annotate_cores(
            aligned_cores, labelMeAnnotations
        )
        self.unmatched_pos_count = 0

    def __get_labelme_core_names(self, labelMeAnnotations):
        labelMeCores = list()
        for shape in labelMeAnnotations['shapes']:
            s = shape['label'].split('_')
            coreName = s[0].upper()
            if coreName not in labelMeCores:
                labelMeCores.append(coreName)
        return labelMeCores

    def __align_core_info(self, labelMeCoreNames, csvCoreNames, csvFirstYears):
        cores = []
        for lm_core in labelMeCoreNames:
            matched = False
            for i, csv_core in enumerate(csvCoreNames):
                if lm_core == csv_core.upper():
                    cores.append((lm_core, csv_core, csvFirstYears[i]))
                    csvCoreNames.pop(i)
                    csvFirstYears.pop(i)
                    matched = True
                    break
            
            if not matched:
                cores.append((lm_core, None, None))
            
            if not matched and csvCoreNames: 
                logging.warn(
                        f"Image {self.image_name}, labelme core {lm_core} did "
                        "not find a matching csv core."
                    )
        if csvCoreNames:
            logging.warn(
                f"Image {self.image_name}, the following csv cores were not "
                f"matched with labelme cores: {csvCoreNames}")
        return cores

    def __annotate_cores(self, cores, labelMeAnnotations):
        core_annos = list()
        for core_upper,_,core_year in cores:
            core_pos_path = self.__get_core_pos_path(core_upper, core_year)
            coreLabelMeAnnotations = self.__processLabelMeAnnos(
                core_upper, labelMeAnnotations
            )
            core_annotation = CoreAnnotation(
                coreLabelMeAnnotations, 
                core_upper, 
                core_pos_path, 
                self.image_path,
                core_year
            )
            core_annos.append(core_annotation)
        return core_annos

    def __get_core_pos_path(self, core, core_year):
        core_pos_path = None
        for file in os.listdir(POINT_LABELS):
            if file[:-4].upper() == core:
                core_pos_path = os.path.join(
                    POINT_LABELS, file
                )
                break
        if core_year is None and core_pos_path is None:
            logging.warn(f"Could not find pos file for core {core}"
                f" in {self.image_path}")
        return core_pos_path

    def __processLabelMeAnnos(self, core, labelMeAnnotations):
        coreShapes = []
        coreLabelMeAnnotations = deepcopy(labelMeAnnotations)
        for shape in coreLabelMeAnnotations["shapes"]:
            shape['label'] = shape['label'].upper()
            if shape['label'].split("_")[0] == core:
                coreShapes.append(shape)
        coreLabelMeAnnotations["shapes"] = coreShapes 
        return coreLabelMeAnnotations

    def __get_image_path(self, labelMeAnnotations):
        return labelMeAnnotations['imagePath']

    def __get_image_height(self, labelMeAnnotations):
        return labelMeAnnotations['imageHeight']
    
    def __get_image_width(self, labelMeAnnotations):
        return labelMeAnnotations['imageWidth']

    def __repr__(self) -> str:
        return (f"Image annotation with JSON: {self.json_path}, "
        f"and cores: {self.cores}")




