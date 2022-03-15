import json

from preprocessing.Core_Annotation import Core_Annotation

# this is a class that reads in a json annotation of an image and creates a class for each core with all relevant
# annotations

class Image_Annotation:
    def __init__(self, json_path, pos_path):
        # read in a json and construct
        f = open(json_path)
        self.pos_path = pos_path
        self.annotations = json.load(f)
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



