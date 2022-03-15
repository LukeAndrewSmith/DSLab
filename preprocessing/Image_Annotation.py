import json
from preprocessing.Core_Annotation import Core_Annotation


class Image_Annotation:
    """
    This is a class that reads in a json annotation created with labelme
    It serves as a gathering point for all cores inside one image
    It contains a collection of all Core_Annotations in the image
    """
    def __init__(self, json_path, pos_path):
        # json_path: path to a json file
        # pos_path: folder that contains all pos files as one json file contains multiple cores
        # needing to access multiple pos files
        # read in a json and construct
        f = open(json_path)
        self.annotations = json.load(f)
        # pos path needs to be given to the Core_Annotations
        self.pos_path = pos_path
        self.cores = self.get_cores()
        self.core_annotations = self.annotate_cores()
        self.image_path = self.get_image_path()

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
            core_annotation = Core_Annotation(self.annotations, core, self.pos_path)
            core_annotations.append(core_annotation)
        return core_annotations

    def get_image_path(self):
        return self.annotations['imagePath']



