##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Performs image alignment - In case of CREDO dataset the aligning is based on translating
# images so that the pixels with highest grayscale intensity will be in the centre of image
# and rotating images so that the brightest collinear pixels will be horizontal.


import math
import numpy as np
from math import atan2
import cv2

import os
from sklearn.decomposition import PCA

# borderMode - border mode in cv2.warpAffine
def align_image_2(img, borderMode = cv2.BORDER_CONSTANT):
    src_copy = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_copy = np.copy(gray)
    my_list = []
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            z = 0
            while z < gray[x,y]:
                my_list.append([y, x])
                z = z + 1

    X = np.array(my_list)
    pca = PCA(n_components=2)
    pca.fit(X)
    mean = pca.mean_
    eigenvectors = pca.components_

    cntr = (int(mean[0]), int(mean[1]))
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = 180 * angle / math.pi

    (cX, cY) = cntr
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    (h, w) = src_copy.shape[:2]

    rotated = cv2.warpAffine(src_copy, M, (w, h), borderMode = borderMode)

    xx = w / 2 - cX
    yy = h / 2 - cY
    M = np.float32([[1, 0, xx], [0, 1, yy]])

    shifted = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]), borderMode = borderMode)

    return shifted

# set data input directory here
input_path = 'd:\\data\\credo\\data'
#output_path = 'd:\\data\\credo\\align_BORDER_REFLECT101'
#output_path = 'd:\\data\\credo\\align_BORDER_CONSTANT'
# set data output directory here
output_path = 'd:\\data\\credo\\align_BORDER_REPLICATE'



input_path_l = len(input_path)
for root, dirs, files in os.walk(input_path):
    for name in files:
        if name.endswith((".png")):
            position = root.index(input_path)
            folder_name = root[input_path_l:]

            dest_folder_name = output_path + folder_name
            if not os.path.exists(dest_folder_name):
                os.makedirs(dest_folder_name)

            print(root + '\\' + name)
            srcI = cv2.imread(root + '\\' + name)
            shifted = align_image_2(srcI, borderMode=cv2.BORDER_REPLICATE)

            shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dest_folder_name  + '\\' +  name, shifted)

