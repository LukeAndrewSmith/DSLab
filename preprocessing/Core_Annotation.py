import os
from preprocessing.geometry import min_bounding_rectangle, mm_to_pixel, pixel_to_mm
import re

class Core_Annotation:
    def __init__(self, annotations, name, pos_path):
        self.name = name
        self.annotations = annotations
        self.pos_path = pos_path
        self.inner_bound = self.get_inner()
        self.outer_bound = self.get_outer()
        self.cracks = self.get_cracks()
        self.bark = self.get_bark()
        self.ctrmid = self.get_ctrmid()
        self.ctrend = self.get_ctrend()
        self.is_tricky = self.get_tricky()
        self.rings, self.dpi, self.pith, self.dist_to_pith, self.years_to_pith = self.get_pos_info()

    def get_inner(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_inner':
                # get polygon
                inner = shape['points']
                # translate to bounding box
                # TODO bugging right now..
                #inner_box = inner
                inner_box = min_bounding_rectangle(inner)
                return inner_box

    def get_outer(self):
        for shape in self.annotations['shapes']:
            if shape['label'] == f'{self.name}_outer':
                # get polygon
                outer = shape['points']
                # translate to bounding box
                # TODO bugging right now...
                #outer_box = outer
                outer_box = min_bounding_rectangle(outer)
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
                for line in f.readlines():
                    print(line)
                    s = list(filter(None,re.split("[ \n;]", line)))
                    if s[0] == '#DPI':
                        dpi = float(s[1])
                    if len(s) > 1:
                        if 'Pith' in s[1]:
                            # looks like this:
                            # #C PithCoordinates=447.146,70.294; DistanceToPith=50.8; YearsToPith=13;
                            pith_mm = s[1].split('=')[1].split(',')
                            # to pixel values:
                            pith = [mm_to_pixel(float(coordinate), dpi) for coordinate in pith_mm]

                            dist_to_pith_mm = s[2].split('=')[1]
                            # to pixel values:
                            dist_to_pith = mm_to_pixel(float(dist_to_pith_mm), dpi)

                            years_to_pith = int(float(s[3].split('=')[1]))
                    if '#' not in s[0] and 'SCALE' not in s[0]:
                        ring_coordinates = s
                        # there can be multiple points on one ring
                        ring = list()
                        for point in ring_coordinates:
                            point = point.split(',')
                            ring_point = [mm_to_pixel(float(coordinate), dpi) for coordinate in point]
                            ring.append(ring_point)
                        rings.append(ring)

        return rings, dpi, pith, dist_to_pith, years_to_pith