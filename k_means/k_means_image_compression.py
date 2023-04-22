import pathlib
import numpy as np
import matplotlib.pyplot as plt
from k_means import (
    find_closest_centroid,
    initialise_centroids,
    run_kmeans, 
    )

"""
TODO:
1. read in image and display
2. if jpg, divide pxl values by 255 to get in range 
3. transform image into 2-D matrix
4. initialise centroids
5. run kmeans to get centroids and allocations per training example
6. show some visualisations
7. get the compressed image 
    - store just the id of the closest centroid per pixel spot
    - use the centroid colors + the idx to get the "recovered" points
    - reshape to the correct dimensions
    - view compressed image

"""
n_colors = 16
img_file_path = "bird_small.png"
img = plt.imread(img_file_path)

# todo: do this with jpg!
# todo: make interactive - args for specifying image path

plt.imshow(img)

if pathlib.Path(img_file_path).suffix == ".jpg":
    img = img / 255

reshaped_img = np.reshape(img, (img.shape[0]*img.shape[1], 3))

initial_centroids = initialise_centroids(reshaped_img, n_colors)

centroids, compressed_img_ids = run_kmeans(reshaped_img, initial_centroids, 10,)

compressed_img_ids.shape
import sys 
compressed_size = sys.getsizeof(compressed_img_ids) + sys.getsizeof(centroids)
original_size = sys.getsizeof(img)

print(f"Original image size: {original_size}")
print(f"Compressed image size: {compressed_size}")
print(f"Number of colors: {n_colors}")

idx = find_closest_centroid(reshaped_img, centroids)
assert idx == compressed_img_ids  # todo: this is not true ?? some error going on...

recovered_img = centroids[idx.astype(int), :]
recovered_img = np.reshape(recovered_img, img.shape)

plt.imshow(recovered_img)
# todo: figure out why plot not opening !

