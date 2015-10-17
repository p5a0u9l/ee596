#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Paul Adams"

from glob import glob
from os.path import join
# import cv2
import matplotlib
from time import time
matplotlib.use("Qt4Agg")
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_similarity_score
import sys
sys.path.append('/home/adamspr/ee596/MachineVision')
from ScreenImage import ScreenImage


def recreate_image(centers, labels, dim):
    d = center.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i, j] = int(center[labels[label_idx]])
            label_idx += 1
    return image


def extract_features(im):
    # image, int --> array, float
    rgb = im.reshape((-1, 3)).astype('float32')
    # sum along columns
    rgbsum = np.sum(rgb, axis=1)
    # prepare for array division
    rgbsum = np.tile(rgbsum, (3, 1)).transpose()
    # avoid div by 0
    rgbsum[np.where(rgbsum == 0)] = 1
    # normalize rgb array
    rgbnorm = np.divide(rgb, rgbsum)
    # create feature vector
    return np.hstack([rgb, rgbnorm])


def main():
    debug = True
    # debug = True
    pngs = glob(join("face_training", "face*.png"))
    si = ScreenImage()

    for png in pngs[:1]:
        im = io.imread(png)

        print("Extracting (r, g) [R, G, B] features... "),
        t0 = time()
        fvec = extract_features(im)
        print("Success. %0.3f sec." % (time() - t0))

        print("Fitting model to features of %s... " % png),
        t0 = time()
        kmeans = KMeans(n_clusters=8, tol=1., max_iter=10).fit(fvec)
        print("Success. %0.3f sec." % (time() - t0))

        print("Predicting color indices of the image... "),
        t0 = time()
        labels = kmeans.predict(fvec)
        print("Success. %0.3f sec." % (time() - t0))

        if debug:
            im3 = recreate_image(kmeans.cluster_centers_, labels, im.shape)
            si.show(im, "Original: " + png)
            si.show(im3, "Kmeans: Quantized " + png)


if __name__ == '__main__':
    main()
