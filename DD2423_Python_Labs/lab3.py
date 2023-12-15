import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from PIL import Image,ImageFilter
from scipy.stats import multivariate_normal


def kmeans_segm(image, K, L, seed = 42, early_stopping=False):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """ 
    # Set random seed
    np.random.seed(seed)

    

    # Initialize centers by sampling K different pixels from the image and using their RGB color as the center
    pixel_values = np.reshape(image, (-1, 3))
    centers = set()
    while len(centers) < K:
        center = pixel_values[np.random.choice(pixel_values.shape[0], 1, replace=False)]
        center = tuple(center[0])
        if center not in centers:
            centers.add(center)
    centers = np.array(list(centers))
    new_centers = np.zeros(centers.shape)
    
    # Check if image is 2D or 3D
    if len(image.shape) == 3:
        image_to_segm = np.reshape(image, (-1, 3))
    else:
        image_to_segm = image

    segmentation = np.zeros(image_to_segm.shape[0]*image_to_segm.shape[1]).astype(np.int32)
    # Compute all distances between pixels and cluster centers
    distances = distance_matrix(image_to_segm, centers)
    # Iterate L times
    for i in range(L):
        # Assign each pixel to the cluster center for which the distance is minimum
        new_segmentation = np.argmin(distances, axis=1)
        
        # Check if segmentation has changed
        if np.array_equal(segmentation, new_segmentation) and early_stopping:
            break
        segmentation = new_segmentation

        # Recompute each cluster center by taking the mean of all pixels assigned to it
        for j in range(K):
            centers[j] = np.mean(image_to_segm[segmentation == j], axis=0)
        
        # Recompute all distances between pixels and cluster centers
        distances = distance_matrix(image_to_segm, centers)
    num_iterations = i + 1 

    if (len(image.shape) == 3):
        segmentation = np.reshape(segmentation, image.shape[:2])

    return segmentation, centers, num_iterations


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """

    image = image / 255
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)
    m = np.reshape(mask, (-1))

    # Remove masked pixels
    Ivec_masked = Ivec[np.reshape(np.nonzero(m == 1), (-1))]
    size = np.shape(Ivec_masked)[0]

    # Compute segmentation and centers with kmeans on masked pixels
    segs, centers,_ = kmeans_segm(Ivec_masked, K, L)
    covariances = np.zeros((K, 3, 3))
    p = np.ones((size, K)) * 0.001
    g = np.zeros((size, K))
    for i in range(K):
        covariances[i]=np.eye(3) * 0.01

    w = np.zeros(K)
    for i in range(K):
        w[i] = len(np.nonzero(segs == i)) / size

    for i in range(L):
        for j in range(K):
            
            g[:, j] = w[j] * multivariate_normal(centers[j], covariances[j]).pdf(Ivec_masked)

        for j in range(K):
            # Expectation step: compute probabilityn for each pixel to belong to each cluster
            p[:, j] = np.divide(g[:, j], np.sum(g, axis=1), where=np.sum(g, axis=1)!=0)

        for j in range(K):
            # Maximization step: update weights, means and covariances
            w[j] = np.mean(p[:,j])
            centers[j,:] = np.dot(np.transpose(p[:, j]), Ivec_masked) / np.sum(p[:, j])
            diff = Ivec_masked - centers[j,:]
            covariances[j] = np.dot(np.transpose(diff), diff * np.reshape(p[:, j], (-1, 1))) / np.sum(p[:, j])

    prob = np.zeros((np.shape(Ivec)[0], K))
    # Compute probability of the color for each pixel
    for i in range(K):
        prob[:, i] = w[i] * multivariate_normal(centers[i], covariances[i]).pdf(Ivec)
        prob[:, i] = prob[:, i] / np.sum(prob[:, i])

    prob = np.sum(prob, axis=1)
    prob = np.reshape(prob, (np.shape(image)[0], np.shape(image)[1]))
    return prob
