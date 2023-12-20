##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Find potential anomalies using density-based search


import numpy as np
img_count = 573335

fileNameSave = 'pca.resBORDER_REFLECT101 550000-560000'
fileNameSave = 'pca.resBORDER_REFLECT101 450000-460000'
fileNameSave = 'pca.resBORDER_REFLECT101 350000-360000'
fileNameSave = 'pca.resBORDER_REFLECT101 250000-260000'
fileNameSave = 'pca.resBORDER_REFLECT101 150000-160000'
fileNameSave = 'pca.resBORDER_REFLECT101 50000-60000'
#fileNameSave = 'pca.resBORDER_REFLECT101'

#fileNameSave = 'pca.resBORDER_REFLECT101'
#fileNameSave = 'pca.resBORDER_CONSTANT'
#fileNameSave = 'pca.resBORDER_REPLICATE'

#fileNameSave = 'pca.res'

change_path = True
change_path = False

if change_path:
    #emb_array = np.load('D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\pca.resBORDER_REFLECT101\\partial3600\\emb_array_wybrane_iterative3600 550000-560000.npy')

    #emb_array = np.load('D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\pca.resBORDER_REFLECT101\\partial3600\\emb_array_wybrane_iterative3600 50000-60000.npy')

    #emb_array = np.load('D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\pca.resBORDER_REFLECT101\\emb_array_wybrane_3600.npy')
    #my_file = open("pca.resBORDER_REFLECT101/image_files_listBORDER_REFLECT101.txt", "r")

    #emb_array = np.load("pca.resBORDER_CONSTANT/emb_array_wybrane_3600.npy")
    #my_file = open("pca.resBORDER_CONSTANT/image_files_listBORDER_CONSTANT.txt", "r")

    emb_array = np.load("pca.resBORDER_REPLICATE/emb_array_wybrane_3600.npy")
    my_file = open("pca.resBORDER_REPLICATE/image_files_listBORDER_REPLICATE.txt", "r")
else:
    emb_array = np.load("pca.res/emb_array_wybrane_3600.npy")
    my_file = open("pca.res/image_files_list.txt", "r")
#my_file = open("pca.res/files.txt", "r")

file_content = my_file.read()
all_files = file_content.split("\n")
emb_array_copy = np.copy(emb_array[0:img_count, 0:62])

eps=2.8
min_samples=3


import numba as nb
@nb.jit(nopython=True)
def func_nb(emb_array_copy, i, eps, min_samples):
    min_samples_help = 0
    x1 = emb_array_copy[i,]
    for j in range(emb_array_copy.shape[0]):
        if i != j:
            x2 = emb_array_copy[j,]
            dist = np.linalg.norm(x1 - x2)
            if dist < eps:
                min_samples_help = min_samples_help + 1
            if min_samples_help > min_samples:
                return False
    return True

my_ids = []
from tqdm import tqdm
for i in tqdm(range(emb_array_copy.shape[0])):
    if func_nb(emb_array_copy, i, eps, min_samples):
        my_ids.append(i)
        print(len(my_ids))

print(my_ids)
import matplotlib.pyplot as plt
import cv2
columns = 8
fig = plt.figure(figsize=(columns, columns))
rows = int(len(my_ids) / columns) + 1

columns2 = columns + 1
rows2 = int(len(my_ids) / columns2) + 1
ret_img = np.zeros((128 * rows2, 128 * columns2, 3))

xx = 0
yy = 0
for aaa in range(len(my_ids)):
    file_name = all_files[my_ids[aaa]]
    if change_path:
        #file_name = file_name.replace('d:\\data\\credo\\align_BORDER_REFLECT101\\', 'd:\\data\\credo\\data\\')
        #file_name = file_name.replace('d:\\data\\credo\\align_BORDER_CONSTANT\\', 'd:\\data\\credo\\data\\')
        file_name = file_name.replace('d:\\data\\credo\\align_BORDER_REPLICATE\\', 'd:\\data\\credo\\data\\')

        my_ids[aaa] = file_name
    else:
        my_ids[aaa] = file_name
    img_help = cv2.imread(my_ids[aaa])
    aaa1 = aaa + 1
    fig.add_subplot(rows, columns, aaa1)

    plt.axis('off')
    plt.tight_layout(pad=0.00)

    plt.imshow(img_help)


print(my_ids)
plt.show()

"""
import os
with open("results//" + fileNameSave + " " + str(eps) + ".txt", 'w') as fp:
    for aaa in range(len(my_ids)):
        file_name = my_ids[aaa]
        file_name = os.path.basename(file_name)
        fp.write(file_name + '\n')
"""

