
# Packages to be imported

import argparse
import utils
import cv2
import matplotlib.pyplot as plt

# KMeans clustering algorithm
from sklearn.cluster import KMeans
'''
 Idea : 1. Clustering the pixel intensities of a RGB image.
        2. Image Size : MxN --> Number of pixels = MxN -->each pixel with RGB component
        3. Clustering of these pixel values using K-means clustering algorithm.
        4. Logic: Pixels that belong to a given cluster will be more similar in color than pixels belonging to a separate cluster
 Issue : Tuning of "k"- Number of clusters as the algorithm is dependent upon this hyperparameter.
'''

# Argument Parser Object
ap = argparse.ArgumentParser()

# Arguments as Command line arguments
ap.add_argument("-i","--image",required = True,help = "Path to the Input Image")
ap.add_argument("-c","--clusters",required = True,help = "Number of Clusters",type = int)

# Parsing the comand line arguments
args = vars(ap.parse_args())

# Image Input, Changing image from BGR to RGB
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Before Reshaping")
print(image.shape)
# 3D - image
# Basic goal : "k" clusters, "n" data points
# Image -> Matrix of pixels
#  For applying clusters to pixel values, we need MxN list of pixels
# MXN -> For list
# 3 for RGB channels
# So, after reshaping the image shape is ->(MxN,3) instead of a matrix
image = image.reshape((image.shape[0]*image.shape[1],3))
print("After Reshaping")
print(image.shape) # 2D - image

# Finding out the most dominant color in the image, forming clusters
clt = KMeans(n_clusters = args["clusters"])

# Fitting the image to the algorithm which means clustering out list of pixels
clt.fit(image)

'''
Logic : 
'''

hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist,clt.cluster_centers_)


plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()




