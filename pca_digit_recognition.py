import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from statistics import mode


data = pd.read_csv('train_small.csv')
test = np.reshape(np.array(data.iloc[6934, 1:]), (28, 28))
print("6934th Image Label:", data.iloc[6934, 0])
plt.figure('6934th Image')
plt.imshow(test, cmap='gray')


# The data points are images, and the things to measure are the individual pixel values
# This is analogous to the student-score problem, where each student's profile is made
# up of their test scores; here, each image is made up of the values of its pixels.
# This makes sense because we want to cluster the images (data points) by the variation
# in the pixel values, which is what PCA allows us to do.


# We should make sure all columns have an average of 0 because we want to
# accurately capture the variance of values for each pixel (one column represents
# the values of a single pixel in all images), so we need to center column values
# on 0 to avoid


data = data.to_numpy()
pixels = data[:, 1:]
pixels = pixels - np.mean(pixels, 0)

# Here, we subtract each column's mean from each column (np.mean, with axis=0
# finds a vector of each column mean, which we can subtract from the pixel values
# to get our centered data). We use columns because we want to center the data around
# each axis. The axes in this case are the 784 pixel values, so to center the data
# on the origin, we must make sure that the values along all these axes are centered.
# Each axis is represented by a column in the data (representing a single pixel position
# in all of the images), so we center our data by subtracting column means from each
# column.


cov = np.cov(pixels.T)
evals, evecs = np.linalg.eig(cov)
eval_i = evals.argsort()[::-1]
evecs = evecs[:, eval_i]
evecs = np.real(evecs)

plt.figure("First 2 PCs as Images")
plt.subplot(1, 2, 1)
plt.imshow(np.reshape(evecs[:, 0], (28, 28)), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(np.reshape(evecs[:, 1], (28, 28)), cmap='gray')

# My first two vectors are given by evecs[:, 0] and evecs[:, 1],
# which are the first two eigenvectors of our eigenvector matrix, sorted
# by decreasing eigenvalue size. The first component has a 0-like shape,
# and the second seems to have a bit more complexity, not resembling any
# digit in particular. In both, the outer edge is blank, which makes sense
# because in all data points, we can observe this pattern.


plt.figure("Data Projected on PCs 1 & 2")

proj_2 = pixels @ evecs[:, :2]

magma = plt.cm.get_cmap('pink', 20)
for i in range(proj_2.shape[0]):
    plt.text(proj_2[i, 0], proj_2[i, 1], str(data[i, 0]), color=magma(data[i, 0]))

plt.xlim([proj_2[:, 0].min(), proj_2[:, 0].max()])
plt.ylim([proj_2[:, 1].min(), proj_2[:, 1].max()])


def proj(k):
    return pixels @ evecs[:, :k]


k_10 = proj(10)
kdt_10 = spatial.KDTree(k_10)
dd, ii = kdt_10.query(k_10, [2])
dist = 10000
for i in range(data.shape[0]):
    indx = ii[i, 0]
    if data[i, 0] != data[indx, 0]:
        dist = min(dist, dd[i][0])
print("Shortest Distance:", dist)



k = 30
proj_30 = proj(k)
avg_proj = []
for i in range(10):
    all_i = proj_30[data[:, 0] == i]
    avg_proj.append(np.mean(all_i, axis=0))

avg_proj = np.array(avg_proj)
plt.scatter(avg_proj[:, 0], avg_proj[:, 1])


def pred_avg(img):
    img_proj = np.dot(img, evecs[:, :k])
    min_d = 100000
    min_l = 0
    for i in range(avg_proj.shape[0]):
        d_new = np.linalg.norm(avg_proj[i] - img_proj)
        if d_new < min_d:
            min_d = min(min_d, d_new)
            min_l = i
    return min_l


print("Predicted 6934th Image:", pred_avg(pixels[6934]))

# The average projection method predicts that the above image is a 3,
# which it is.


hit = 0
for i in range(pixels.shape[0]):
    pred = pred_avg(pixels[i])
    real = data[i, 0]
    if real == pred:
        hit += 1

print("Average Projection Accuracy:", hit / pixels.shape[0])

# Using the average projection method, the model was correct on 80% of the data.


kdt = spatial.KDTree(proj_30)
dd, ii = kdt.query(proj_30, range(2, 7))
dist = 10000
preds = []
for i in range(data.shape[0]):
    labs = []
    for j in range(5):
        labs.append(data[ii[i, j], 0])
    preds.append(mode(labs))

print("Near Projection Accuracy:", np.sum(preds == data[:, 0]) / data.shape[0])

# Using the nearest neighbors method, 97% of the data is predicted
# correctly.


def pred_near(img):
    img_real = np.dot(img, evecs[:, :k])

    dists = np.linalg.norm(proj_30 - img_real, axis=1)
    n = 5
    bot_5 = np.argpartition(dists, n)[:n]
    return mode(data[bot_5, 0])


img = (1 - plt.imread('img.png')[:, :, 0]) * 255
plt.figure("Drawn Image")
plt.imshow(img, cmap='gray')
print("Average Projection Prediction:", pred_avg(img.flatten()))
print("Nearest Neighbor Prediction:", pred_near(img.flatten()))

# The image 'img.png' is correctly identified as a 4 by the nearest
# neighbor method, and incorrectly identified by the average projection
# method.


img = (1 - plt.imread('img_1.png')[:, :, 0]) * 255
plt.figure("Drawn Image 2")
plt.imshow(img, cmap='gray')
print("Nearest Neighbor Prediction:", pred_near(img.flatten()))

# Nearest neighbors incorrectly predicts that 'img_1.png' is 0

plt.show()
