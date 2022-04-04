#%%
from copy import deepcopy
from ringdetector.analysis.ImageProcessor import ImageProcessor
from ringdetector.analysis.EdgeProcessor import EdgeProcessor
from ringdetector.Paths import GENERATED_DATASETS_INNER_CROPS, DATA
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#%%
imgpath = os.path.join(GENERATED_DATASETS_INNER_CROPS,"KunA08NN.jpg")

gs_img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
gbr_img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
hsv_img = cv2.cvtColor(gbr_img, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv",hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

v_img = hsv_img[:,:,2]

cv2.imshow("v",v_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


denoise_10 = cv2.fastNlMeansDenoising(
    v_img, None, 10
)

denoise_5 = cv2.fastNlMeansDenoising(
    v_img, None, 5
)

denoise_15 = cv2.fastNlMeansDenoising(
    v_img, None, 15
)
  
# concatanate image Vertically
Verti = np.concatenate([denoise_5, denoise_10, denoise_15], axis=0)[:,:2000]
  
cv2.imshow('VERTICAL', Verti)

cv2.waitKey(0)
cv2.destroyAllWindows()

g1 = cv2.Sobel(denoise_10, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
g2x = cv2.Sobel(denoise_10, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
g2x = cv2.Sobel(denoise_10, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
g2y = cv2.Sobel(denoise_10, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
mag, angle = cv2.cartToPolar(g2x, g2y, angleInDegrees=True)

Verti = np.concatenate([g1,g2], axis=0)[:,:2000]

gradImage = Image.fromarray(g1[:,:2000].astype(np.uint8))

cv2.imshow('signed', g2[:,2000:4000])

cv2.waitKey(0)
cv2.destroyAllWindows()

can = cv2.Canny(v_img, threshold1=50, threshold2=100)
cans = [
    v_img,
    cv2.Canny(v_img, threshold1=40, threshold2=80),
    cv2.Canny(denoise_10, threshold1=40, threshold2=80),
    cv2.Canny(denoise_10, threshold1=50, threshold2=80),
    cv2.Canny(denoise_10, threshold1=60, threshold2=80)
]
Verti = np.concatenate(cans, axis=0)[:,13000:15000]
cv2.imshow('can', Verti)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% START HERE
testSample = "KunA01SS"
proc = ImageProcessor(
    os.path.join(
        GENERATED_DATASETS_INNER_CROPS, f"{testSample}.jpg"
    ),
    "hsv"
)
proc.computeGradients(method="Canny", threshold1=40, threshold2=80)

#%%
cans = [proc.denoisedImage, proc.gXY]
Verti = np.concatenate(cans, axis=0)[:,:3000]
cv2.imshow('can', Verti)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
ep = EdgeProcessor(proc.gXY)
ep.processEdgeInstances(196/2)
im = ep.ImageFromEdgeInstances("processed").astype(np.uint8)

#%%
cans = [proc.denoisedImage, proc.gXY, im]
Verti = np.concatenate(cans, axis=0)[:,:3000]
cv2.imshow('can', Verti)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

from sklearn.linear_model import LinearRegression

shapes = deepcopy(ep.processedEdges)
shapes.sort(key=len, reverse=True)

shape = shapes[20]

y = [point[1] for point in shape]
X = [point[0] for point in shape]
horiz_min = min(y)
horiz_max = max(y)

imgheight, imgwidth = proc.denoisedImage.shape
shapeim = np.zeros((imgheight, imgwidth, 3), dtype=np.uint8)
for point in shape:
    shapeim[point] = (255,255,255)

#%%

cv2.imshow('can', shapeim[:,horiz_min-60:horiz_max+60])

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
cans = [proc.denoisedImage, im, shapeim]
Verti = np.concatenate(cans, axis=0)[:,horiz_min-60:horiz_max+60]
cv2.imshow('can', Verti)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
lm = LinearRegression()
lm.fit(np.array(X).reshape(-1, 1),np.array(y))

pred_domain = np.arange(imgheight)
pred = np.rint(
        lm.predict(pred_domain.reshape(-1, 1))
    ).astype(np.int)

pred_coords = list(zip(pred_domain, pred))

for coord in pred_coords:
    shapeim[coord[0], coord[1],:] = (0,0,255)

# %%
cv2.imshow('can', shapeim[:,horiz_min-60:horiz_max+60])

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
from ringdetector.Paths import GENERATED_DATASETS_INNER_PICKLES
import pickle 

picklePath = os.path.join(GENERATED_DATASETS_INNER_PICKLES, f"{testSample}.pkl")
with open(picklePath, "rb") as f:
    core = pickle.load(f)

points = [
    [[round(coord) for coord in coords]
        for coords in shape] for shape in core.pointLabels
]

shapeim[points[32][0][1], points[32][0][0], :] = (0,255,0)

# %%
import math

flattened_points = [item for sublist in points for item in sublist]

def pixelDist(a, b):
    dx2 = (a[0]-b[0])**2
    dy2 = (a[1]-b[1])**2
    return math.sqrt(dx2 + dy2)

def findClosestLabel(flattened_points, pred_coords):
    min_dist = 100000
    min_label_point = (0,0)
    for point in flattened_points:
        flipped_point = (point[1], point[0])
        for pred in pred_coords:
            dist = pixelDist(flipped_point, pred)
            if dist > 1000 and dist > min_dist:
                break
            else:
                if dist < min_dist:
                    min_dist = dist
                    min_label_point = flipped_point
    return min_dist, min_label_point
        
# %%
