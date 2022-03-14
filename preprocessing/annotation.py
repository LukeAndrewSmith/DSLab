import json
from geometry import min_bounding_rectangle

# this is a class that reads in a json annotation of an image and creates a class for each core with all relevant
# annotations

class Image_Annotation:
    def __init__(self, json_path):
        # read in a json and construct
        f = open(json_path)
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
            core_annotation = Core_Annotation(self.annotations, core)
            core_annotations.append(core_annotation)
        return core_annotations

    def get_image_path(self):
        return self.annotations['imagePath']


class Core_Annotation:
    def __init__(self, annotations, name):
        self.name = name
        self.annotations = annotations
        self.inner = self.get_inner()
        self.outer = self.get_outer()
        self.cracks = self.get_cracks()
        self.bark = self.get_bark()
        self.ctrmid = self.get_ctrmid()
        self.ctrend = self.get_ctrend()
        self.is_tricky = self.get_tricky()

    def get_inner(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_inner':
                # get polygon
                inner = shape['points']
                # translate to bounding box
                # TODO bugging right now..
                inner_box = inner
                #inner_box = min_bounding_rectangle(inner)
                return inner_box

    def get_outer(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_outer':
                # get polygon
                outer = shape['points']
                # translate to bounding box
                # TODO bugging right now...
                outer_box = outer
                # outer_box = min_bounding_rectangle(outer)
                return outer_box

    def get_cracks(self):
        cracks = list()
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_crack':
                # get polygon
                crack = shape['points']
                cracks.append(crack)
        return cracks

    def get_bark(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_bark':
                # get polygon
                bark = shape['points']
                return bark

    def get_ctrmid(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_ctrmid':
                # get polygon
                ctrmid = shape['points']
                return ctrmid

    def get_ctrend(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_ctrend':
                # get polygon
                ctrend = shape['points']
                return ctrend

    def get_tricky(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_tricky':
                return True
        return False


if __name__ == "__main__":
    json_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels/KunA08.json"
    annotation = Image_Annotation(json_path)
    print(annotation)
