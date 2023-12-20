##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Generate embedding of aligned dataset (BORDER_REFLECT) from incremental eigendecomposition algorithm

import numpy as np

how_many_images = 573335

path_to_results = "pca.resBORDER_REFLECT101//"

image_files_list = "image_files_listBORDER_REFLECT101.txt"
path = 'd:\\data\\credo\\align_BORDER_REFLECT101\\'
path_to_resultspartial = "pca.resBORDER_REFLECT101\\partial3600\\"

v_correct = np.load(path_to_results + "//v_st_" + str(how_many_images) + ".npy")
w = np.load(path_to_results + "//w_st_" + str(how_many_images) + ".npy")
mean_face = np.load(path_to_results + "//mean_face_st_" + str(how_many_images) + ".npy")
norms = np.load(path_to_results + "//norms_st_" + str(how_many_images) + ".npy")
old_shape = np.load(path_to_results + "//old_shape_st_" + str(how_many_images) + ".npy")
how_many_images = v_correct.shape[1]

##################################################
#start = 550000
#end =  560000
#start = 450000
#end =  460000
#start = 350000
#end =  360000
start = 50000
end =  60000

import pickle
transformer = None
my_path = path_to_resultspartial + 'pickle' + str(start) + '-' + str(end) + '.pkl'
with open(my_path, 'rb') as inp:
    transformer = pickle.load(inp)

v_correct_2 = transformer.components_.transpose()
v_correct = v_correct_2
mean_face = transformer.mean_

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
from tqdm import tqdm
T = np.load(path_to_results + "//full_data.npy").transpose()
#for file in all_files:
for a in tqdm(range(T.shape[0])):
    img_help = T[a,:]
    embed = embedding(img_help, v_correct, mean_face)
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

np.save(path_to_resultspartial + "//emb_array_wybrane_iterative" + str(how_many_images) + " " + str(start) + '-' + str(end), emb_array)
