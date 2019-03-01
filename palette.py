# design an algorithm that pre-determines the optimal k

# goal is to generate k cluters of n data points

from sklearn.cluster import KMeans
from matplotlib import image as img
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())

# plot representing 3 rgb values on the corresponding axes
graph = img.imread('./wave.jpeg')
r = []
g = []
b = []

for line in graph:
    for pixel in line:
        temp_r,temp_g,temp_b = pixel
        r.append(temp_r)
        b.append(temp_b)
        g.append(temp_g)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(r,g,b)
plt.show()

# load the image
image = cv2.imread(args["image"])
# convert it from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

# display colors
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
