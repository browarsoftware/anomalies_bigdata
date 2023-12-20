##########################################
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2023
##########################################
# Generate eigendecomposition of various dataset with Incremental PCA

import cv2
import numpy as np
import os



"""
path = 'd:\\data\\credo\\align_BORDER_REFLECT101\\'
path_to_results = "pca.resBORDER_REFLECT101\\"
path_to_resultspartial = "pca.resBORDER_REFLECT101\\partial3600\\"
"""

path = 'd:\\data\\credo\\align_BORDER_REPLICATE\\'
path_to_results = "pca.resBORDER_REPLICATE\\"
path_to_resultspartial = "pca.resBORDER_REPLICATE\\partial3600\\"

"""
path = 'd:\\data\\credo\\align_BORDER_CONSTANT\\'
path_to_results = "pca.resBORDER_CONSTANT\\"
path_to_resultspartial = "pca.resBORDER_CONSTANT\\partial3600\\"
"""

"""
path = 'd:\\data\\credo\\data\\'
path_to_results = "pca.res\\"
path_to_resultspartial = "pca.res\\partial3600\\"
"""



#image_files_list = "image_files_listBORDER_REPLICATE.txt"
image_files_list = "image_files_listBORDER_REPLICATE.txt"
#image_files_list = "image_files_list.txt"

with open(path_to_results + "//" + image_files_list, 'r') as fp:
    all_files = fp.readlines()

how_many_images = len(all_files)

full_path = os.path.join(str.strip(all_files[0]))
img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

old_shape = img.shape
img_flat = img.flatten('F')

"""
T = np.zeros((img_flat.shape[0], len(all_files)))
for i in range(len(all_files)):
#for i in range(30000):
    if i % 1000 == 0:
        print("\tLoading " + str(i) + " of " + str(len(all_files)))
    full_path = os.path.join(str.strip(all_files[i]))
    img_help = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    #img_help = cv2.imread(all_files[i], cv2.IMREAD_GRAYSCALE)
    T[:,i] = img_help.flatten('F') / 255

np.save(path_to_results + "//full_data", T)
"""

T = np.load(path_to_results + "//full_data.npy").transpose()

batch_size=10000
start = 0
end = 0

import pickle
from sklearn.decomposition import IncrementalPCA
transformer = IncrementalPCA(n_components=T.shape[1], batch_size=batch_size)

a = 0

while end < T.shape[0]:
    end = start + batch_size
    if end > T.shape[0]:
        end = T.shape[0]
    bb = T[start:end:]
    transformer.partial_fit(bb)

    my_path = path_to_resultspartial + 'pickle' + str(start) + '-' + str(end) + '.pkl'
    with open(my_path, 'wb') as outp:
        pickle.dump(transformer, outp, pickle.HIGHEST_PROTOCOL)
    transformer = None

    with open(my_path, 'rb') as inp:
        transformer = pickle.load(inp)

    start = end
    print(str(a) + " of " + str(int(T.shape[0] / batch_size)))
    a = a + 1
