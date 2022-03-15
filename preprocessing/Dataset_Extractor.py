import os
from Image_Annotation import Image_Annotation
from Core_Annotation import Core_Annotation
from preprocessing.geometry import mm_to_pixel, pixel_to_mm

class Dataset_Extractor:
# Class to extract datasets from image and core annotations

    def __init__(self):
        self._initCoreAnnotations()

    def _initCoreAnnotations(self):
        self.coreAnnotations = []
        dir = 'data/labelme_jsons/'
        for file in os.listdir(dir):
            self.coreAnnotations = self.coreAnnotations + Image_Annotation(dir+file, 'data/point_labels').core_annotations

    def createInnerDataset(self):
        self._apply(self._processCore, self.coreAnnotations)

    def _processCore(self, core):
        # Convert mm->px
        # Shift points
        # Rotate rectangle/points
        # Crop image
        # Save image and pos file
        self._shiftPoints(core)
        self._rotateImageAndPoints()
        self._cropImage()
        self._saveImage()

    def _shiftPoints(self):
        pass

    def _rotateImageAndPoints(self):
        pass

    def _cropImage(self):
        pass

    def _saveImage(self):
        pass

    ###############
    # Helpers
    def _apply(self, func, list):
        for item in list:
            func(item)

# USEFUL CODE FOR CONVERTING MM TO PX
# def get_pos_info(self):
    #     rings = list()
    #     dpi = 0
    #     pith = None
    #     dist_to_pith = None
    #     years_to_pith = None
    #     for file in os.listdir(self.pos_path):
    #         if file == f'{self.name}.pos':
    #             f = open(os.path.join(self.pos_path, file))
    #             for line in f.readlines():
    #                 s = list(filter(None,re.split("[ \n;]", line)))
    #                 if s[0] == '#DPI':
    #                     dpi = float(s[1])
    #                 if len(s) > 1:
    #                     if 'Pith' in s[1]:
    #                         # looks like this:
    #                         # #C PithCoordinates=447.146,70.294; DistanceToPith=50.8; YearsToPith=13;
    #                         pith_mm = s[1].split('=')[1].split(',')
    #                         # to pixel values:
    #                         pith = [mm_to_pixel(float(coordinate), dpi) for coordinate in pith_mm]

    #                         dist_to_pith_mm = s[2].split('=')[1]
    #                         # to pixel values:
    #                         dist_to_pith = mm_to_pixel(float(dist_to_pith_mm), dpi)

    #                         years_to_pith = int(float(s[3].split('=')[1]))
    #                 if '#' not in s[0] and 'SCALE' not in s[0]:
    #                     ring_coordinates = s
    #                     # there can be multiple points on one ring
    #                     ring = list()
    #                     for point in ring_coordinates:
    #                         point = point.split(',')
    #                         ring_point = [mm_to_pixel(float(coordinate), dpi) for coordinate in point]
    #                         ring.append(ring_point)
    #                     rings.append(ring)

    #     return rings, dpi, pith, dist_to_pith, years_to_pith

