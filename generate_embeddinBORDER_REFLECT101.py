##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Generate embedding of aligned dataset (BORDER_REFLECT)


import cv2
import numpy as np


how_many_images = 573335
path_to_results = "pca.resBORDER_REFLECT101//"

image_files_list = "image_files_listBORDER_REFLECT101.txt"
path = 'd:\\data\\credo\\align_BORDER_REFLECT101\\'

v_correct = np.load(path_to_results + "//v_st_" + str(how_many_images) + ".npy")
w = np.load(path_to_results + "//w_st_" + str(how_many_images) + ".npy")
mean_face = np.load(path_to_results + "//mean_face_st_" + str(how_many_images) + ".npy")
norms = np.load(path_to_results + "//norms_st_" + str(how_many_images) + ".npy")
old_shape = np.load(path_to_results + "//old_shape_st_" + str(how_many_images) + ".npy")
how_many_images = v_correct.shape[1]


def embedding(carrier_img_i, v, mean_face):
    carrier_img = np.copy(carrier_img_i)
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    result = np.matmul(v.transpose(), img_flat)
    return result


all_embedding = []
all_files = []

with open(path_to_results + "//" + image_files_list, 'r') as fp:
    all_files = fp.readlines()

import os
# r=root, d=directories, f = files
for file in all_files:
    full_path = os.path.join(str.strip(file))
    img_help = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    off = 0
    embed = embedding(img_help / 255, v_correct, mean_face)
    all_embedding.append(embed)


emb_array = np.zeros((len(all_embedding), len(all_embedding[0])))
a = 0

with open(path_to_results + '//files.txt', 'w') as f:
    for emb in all_embedding:
        emb_array[a, :] = emb
        my_str = all_files[a].strip()
        my_str = os.path.basename(my_str)
        f.write(my_str + "\n")
        a = a + 1
np.save(path_to_results + "//emb_array_wybrane_" + str(how_many_images), emb_array)