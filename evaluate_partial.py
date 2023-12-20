##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Comparison of coordinate frames obtained with basic PCA to coordinate frames obtained with Incremental PCA
# for a different number of data used when approximating PCA

import numpy as np
import pickle

def correct_vector(v1):
    v_help = np.copy(v1)
    max_val = np.max(v_help)
    min_val = np.min(v_help)
    if np.abs(min_val) > np.abs(max_val) and min_val < 0:
        v_help *= -1
    return v_help

import math
def planar_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if math.isnan(angle):
        angle = 0
    return angle

"""
path_to_results = "pca.resBORDER_REFLECT101//"
path_to_resultspartial = "pca.resBORDER_REFLECT101\\partial3600\\"
file = open('results/axes_similarityBORDER_REFLECT101.txt', 'w')
"""

"""
path_to_results = "pca.resBORDER_REPLICATE//"
path_to_resultspartial = "pca.resBORDER_REPLICATE\\partial3600\\"
file = open('results/axes_similarityBORDER_REPLICATE.txt', 'w')
"""

"""
path_to_results = "pca.resBORDER_CONSTANT//"
path_to_resultspartial = "pca.resBORDER_CONSTANT\\partial3600\\"
file = open('results/axes_similarityBORDER_CONSTANT.txt', 'w')
"""

# path to results computed with PCA
path_to_results = "pca.res//"
# path to results computed with Incremental PCA
path_to_resultspartial = "pca.res\\partial3600\\"
# path to store results
file = open('results/axes_similarityPCA.txt', 'w')

#start = 560000
#end =  570000
start = 0
end =  10000
how_many_images = 573335

v_correct = np.load(path_to_results + "//v_st_" + str(how_many_images) + ".npy")
w = np.load(path_to_results + "//w_st_" + str(how_many_images) + ".npy")
v_correct_scalled = w / np.sum(w)

transformer = None

for ss in range(0, 580000, 10000):
    start = ss
    end = start + 10000
    my_path = path_to_resultspartial + 'pickle' + str(start) + '-' + str(end) + '.pkl'
    with open(my_path, 'rb') as inp:
        transformer = pickle.load(inp)

    cumulatice_sum = 0
    sum_angle_res = 0
    # calculation of coordinate frames weighted distance (cfd)
    for a in range(v_correct.shape[1]):
        v_help1 = correct_vector(v_correct[:, a])
        v_help2 = correct_vector(transformer.components_[a, :])
        angle_res = planar_angle(v_help1, v_help2)
        average_eigen = (v_correct_scalled[a] + transformer.explained_variance_ratio_[a]) / 2
        cumulatice_sum += average_eigen
        sum_angle_res += average_eigen * angle_res
    print(str(start) + "," + str(end) + "," + str(sum_angle_res))
    file.write(str(start) + "," + str(end) + "," + str(sum_angle_res) + '\n')

file.close()