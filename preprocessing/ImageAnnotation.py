import json
import os
import logging
from preprocessing.CoreAnnotation import CoreAnnotation

# this is a class that reads in a json annotation of an image and creates a class for each core with all relevant
# annotations

class ImageAnnotation:
    def __init__(self, json_path, pos_path):
        # read in a json and construct
        self.json_path = json_path
        with open(self.json_path) as f:
            self.pos_path = pos_path
            self.annotations = json.load(f)
            self.image_path = self.get_image_path()
            self.cores = self.get_cores()
            self.core_annotations = self.annotate_cores()
            self.unmatched_pos_count = 0

    def get_cores(self):
        cores = list()
        for shape in self.annotations['shapes']:
            s = shape['label'].split('_')
            if s[0] not in cores:
                cores.append(s[0])
        return cores

    def annotate_cores(self):
        core_annotations = list()
        for core in self.cores:
            core_pos_path = self._get_core_pos_path(core)
            #TODO: could be cleaner to filter to annotations only for the specific core before passing to Core_Annotation
            if core_pos_path is not None:
                core_annotation = CoreAnnotation(
                    self.annotations, 
                    core, 
                    core_pos_path, 
                    self.image_path
                )
            core_annotations.append(core_annotation)
        return core_annotations

    def _get_core_pos_path(self, core):
        core_pos_path = os.path.join(
            self.pos_path, core+".pos"
        )
        if os.path.exists(core_pos_path):
            return core_pos_path
        else:
            logging.warn(f"Could not find pos file for core {core}"
            f" in {self.image_path}")
            return None

    def get_image_path(self):
        return self.annotations['imagePath']

    def __repr__(self) -> str:
        return (f"Image annotation with JSON: {self.json_path}, "
        f"and cores: {self.cores}")




