## Requirements

Tested on Python = 3.8; numba = 0.5; numpy = 1.22, opencv-python = 4.5, scikit-learn = 1.0, scipy = 1.8

R Language = 3.6.2; ggplot2 = 3.4

## How to run

1. Download dataset and unpack it;
2. Set path in all *.py files to unpacked data, set path to results data;
3. Run script [align_CREDO_images.py](align_CREDO_images.py), you can set four types of embedding;
4. Run scripts to generate embedding using eigendecompostion with PCA:
	- [generate_eigendecomposition.py](generate_eigendecomposition.py)
	- [generate_eigendecomposition_BORDER_CONSTANT.py](generate_eigendecomposition_BORDER_CONSTANT.py)
	- [generate_eigendecomposition_BORDER_REPLICATE.py](generate_eigendecomposition_BORDER_REPLICATE.py)
	- [generate_eigendecomposition_BORDER_REFLECT101.py](generate_eigendecomposition_BORDER_REFLECT101.py)
5. Run script to generate embedding using eigendecompostion with Incremental PCA:
	- [generate_eigendecomposition_incremental.py](generate_eigendecomposition_incremental.py), you can set four types of embedding;
6. Generate embedding of the dataset using eigendecompostion generated with PCA:
	- [generate_embedding.py](generate_embedding.py)
	- [generate_embeddinBORDER_CONSTANT.py](generate_embeddinBORDER_CONSTANT.py)
	- [generate_embeddinBORDER_REPLICATE.py](generate_embeddinBORDER_REPLICATE.py)
	- [generate_embeddinBORDER_REFLECT101.py](generate_embeddinBORDER_REFLECT101.py)
7. Generate embedding of the dataset using eigendecompostion generated with Incremental PCA:
	- [generate_embeddinBORDER_REFLECT101_incremental.py](generate_embeddinBORDER_REFLECT101_incremental.py), you can set four types of embedding 
8. Run script to find potential anomalies using density-based search [find_anomalies_density.py](find_anomalies_density.py), you can choose from various embedding;
9. Run script to query the object database for the most similar objects [find_most_similar.py](find_most_similar.py), you can choose from various embedding
10. Plot results [plots.R](plots.R) you have to set correct path to results.

	

## Data to download

Download dataset with 573335 images from: [CREDO dataset (1014MB)](https://drive.google.com/file/d/1jSuQXfxFzWsFoTEYDno1V_Aqn5AaNs_I/view)