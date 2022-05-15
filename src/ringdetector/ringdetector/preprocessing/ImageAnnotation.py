import json
import os
import logging
from copy import deepcopy

from ringdetector.preprocessing.CoreAnnotation import CoreAnnotation


# this is a class that reads in a json annotation of an image and creates a class for each core with all relevant
# annotations

class ImageAnnotation:
    def __init__(self, json_path, pos_path):
        # read in a json and construct
        self.json_path = json_path
        with open(self.json_path) as f:
            self.pos_path = pos_path
            self.labelMeAnnotations = json.load(f)
            self.image_path = self.__get_image_path()
            self.cores = self.__get_cores()
            self.core_annotations = self.__annotate_cores()
            self.unmatched_pos_count = 0

    def __get_cores(self):
        cores = list()
        for shape in self.labelMeAnnotations['shapes']:
            s = shape['label'].split('_')
            coreName = s[0].upper()
            if coreName not in cores:
                cores.append(coreName)
        return cores

    def __annotate_cores(self):
        core_annos = list()
        for core in self.cores:
            core_pos_path = self.__get_core_pos_path(core)
            coreLabelMeAnnotations = self.__processLabelMeAnnos(core)
            core_annotation = CoreAnnotation(
                coreLabelMeAnnotations, 
                core, 
                core_pos_path, 
                self.image_path
            )
            core_annos.append(core_annotation)
        return core_annos

    def __get_core_pos_path(self, core):
        core_pos_path = ""
        for file in os.listdir(self.pos_path):
            if file[:-4].upper() == core:
                core_pos_path = os.path.join(
                    self.pos_path, file
                )
                break
        if core_pos_path and os.path.exists(core_pos_path):
            return core_pos_path
        else:
            # TODO: too verbose for justine
            # logging.warn(f"Could not find pos file for core {core}"
            # f" in {self.image_path}")
            return None

    def __processLabelMeAnnos(self, core):
        coreShapes = []
        coreLabelMeAnnotations = deepcopy(self.labelMeAnnotations)
        for shape in coreLabelMeAnnotations["shapes"]:
            shape['label'] = shape['label'].upper()
            if shape['label'].split("_")[0] == core:
                coreShapes.append(shape)
        coreLabelMeAnnotations["shapes"] = coreShapes 
        return coreLabelMeAnnotations

    def __get_image_path(self):
        return self.labelMeAnnotations['imagePath']

    def __repr__(self) -> str:
        return (f"Image annotation with JSON: {self.json_path}, "
        f"and cores: {self.cores}")




