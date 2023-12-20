##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Querying the object database for the most similar
import numpy as np
img_count = 573335

change_path = True
# load embedded data
emb_array = np.load("pca.resBORDER_REFLECT101/emb_array_wybrane_3600.npy")
#emb_array = np.load('D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\pca.resBORDER_REFLECT101\\partial3600\\emb_array_wybrane_iterative3600 550000-560000.npy')
# path to file names
my_file = open("pca.resBORDER_REFLECT101/image_files_listBORDER_REFLECT101.txt", "r")

file_content = my_file.read()
all_files = file_content.split("\n")
emb_array_copy = np.copy(emb_array[0:img_count, 0:62])


dist = np.zeros(emb_array_copy.shape[0])

# set a list of images to which most similar will be found
my_id_list = [1, 95791, 470976, 35073, 70499, 156363, 461992, 296866, 426999]

from tqdm import tqdm
def find_most_similar_from_list(x1):
    for i in tqdm(range(emb_array_copy.shape[0])):
        dist[i] = np.linalg.norm(x1 - emb_array_copy[i,])

    indices = np.argsort(dist)
    dist_sort = dist[indices]
    return (indices, dist_sort)

my_ids = []
my_dist = []
for my_id in my_id_list:
    (indices, dist_sort) = find_most_similar_from_list(emb_array_copy[my_id,])
    _my_ids = indices[0:8]
    _my_dist = dist_sort[0:8]
    my_ids.extend(_my_ids.tolist())
    my_dist.extend(_my_dist.tolist())

import matplotlib.pyplot as plt
import cv2
columns = 8
fig = plt.figure(figsize=(len(my_id_list), columns))


xx = 0
yy = 0
for aaa in range(len(my_ids)):
    file_name = all_files[my_ids[aaa]]
    # path mapping to show not original images, not aligned ones
    if change_path:
        file_name = file_name.replace('d:\\data\\credo\\align_BORDER_REFLECT101\\', 'd:\\data\\credo\\data\\')
    all_files[my_ids[aaa]] = file_name
    img_help = cv2.imread(all_files[my_ids[aaa]])
    aaa1 = aaa + 1
    fig.add_subplot(len(my_id_list), columns, aaa1)
    plt.axis('off')
    plt.tight_layout(pad=0.00)
    plt.imshow(img_help)

print(my_ids)
print(my_dist)
plt.show()

