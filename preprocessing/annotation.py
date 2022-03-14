import json
from geometry import min_bounding_rectangle, mm_to_pixel, pixel_to_mm
import os

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

    def convert_image(self):
        # this function takes in the annotated image and produces an inner and an outer image
        # it also produces new gt labels from the .pos files
        pass

class Core_Annotation:
    def __init__(self, annotations, name, pos_path):
        self.name = name
        self.annotations = annotations
        self.pos_path = pos_path
        self.inner = self.get_inner()
        self.outer = self.get_outer()
        self.cracks = self.get_cracks()
        self.bark = self.get_bark()
        self.ctrmid = self.get_ctrmid()
        self.ctrend = self.get_ctrend()
        self.is_tricky = self.get_tricky()
        self.rings = self.get_pos_info()

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

    def get_pos_info(self):
        rings = list()
        dpi = 0
        pith = None
        dist_to_pith = None
        years_to_pith = None
        for file in os.listdir(self.pos_path):
            if file == f'{self.name}.pos':
                f = open(os.path.join(self.pos_path, file))
                for line in f.readline():
                    s = line.split(' ')
                    if s[0] == '#DPI':
                        dpi = int(s[1])
                    elif 'Pith' in s[1]:
                        # looks like this:
                        # #C PithCoordinates=447.146,70.294; DistanceToPith=50.8; YearsToPith=13;
                        pith_mm = s[1].split('=')[1].split(',')
                        # to pixel values:
                        pith = [mm_to_pixel(int(coordinate), dpi) for coordinate in pith_mm]

                        dist_to_pith_mm = s[2].split('=')[1]
                        # to pixel values:
                        dist_to_pith = mm_to_pixel(dist_to_pith_mm, dpi)

                        years_to_pith = s[3].split('=')[1]
                    elif '#' not in s[0] and 'SCALE' not in s[0]:
                        ring_coordinates = s[0].split(' ')

                        ring = [mm_to_pixel(int(coordinate)) for coordinate in ring_coordinates]
                        rings.append(ring)





if __name__ == "__main__":
    pos_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/Data label pos files"
    json_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels/KunA08.json"
    annotation = Image_Annotation(json_path, pos_path)
    print(annotation)
