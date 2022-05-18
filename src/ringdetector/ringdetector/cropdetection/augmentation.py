import detectron2.data.transforms as T

class RatioResize(T.Augmentation):
    def __init__(self, ratio):
        self.ratio = ratio
    
    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        
        new_h = self.ratio * old_h
        new_w = self.ratio * old_w
        
        return T.ResizeTransform(int(old_h), int(old_w), int(new_h), int(new_w))