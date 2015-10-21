#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Paul Adams"
import matplotlib
matplotlib.use("Qt4Agg")
import re
import matplotlib.pyplot as plt
from glob import glob
from os.path import join, split
import numpy as np
from skimage.filters import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import jaccard_similarity_score
from scipy.ndimage.filters import median_filter
from time import time
import sys
import ipdb
sys.path.append('/home/adamspr/ee596/MachineVision')
from ScreenImage import ScreenImage


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
    return np.hstack([rgb, rgbnorm[:, :2]])


def get_majority_label(kmeans, im_lb, im_tr):
    w, h = im_lb.shape
    im_lb += 1  # prevent finding "0" labels
    # intersection of label and truth
    im_mask = im_lb * im_tr
    # Count occurences of each label in intersection image
    label_counts = [np.count_nonzero(im_mask == i + 1)
                    for i in range(kmeans.n_clusters)]
    major_label = np.argmax(label_counts)
    # RGB values of major label
    major_values = kmeans.cluster_centers_[major_label][:3]
    # create binary image of major label
    im_out = np.zeros((w, h, 3), np.float32)
    im_lbl = np.zeros((w, h), np.uint8)
    for i in range(w):
        for j in range(h):
            if im_lb[i, j] == major_label + 1:
                im_out[i, j, :] = major_values
                im_lbl[i, j] = 1
    # return index of majority label and image
    return major_label, im_out, im_lbl


def log_opponent(im):
    # from forsyth, skin_finding
    im[im == 0] = 0.01  # avoid NaN
    # The input R G B values are transformed into a log opponent representation
    i = np.log(im[:, :, 1])
    rg = np.log(im[:, :, 0]) - i
    by = np.log(im[:, :, 2]) - (i + rg) / 2

    # "The Rg and By arrays are smoothed with a median filter"
    rg = median_filter(rg, 4)
    by = median_filter(by, 4)

    i = median_filter(i, 4)
    imdiff = im
    imdiff[:, :, 0] = np.abs(im[:, :, 0] - i)
    imdiff[:, :, 1] = np.abs(im[:, :, 1] - i)
    imdiff[:, :, 2] = np.abs(im[:, :, 2] - i)
    im = np.zeros((im.shape[0], im.shape[1], 5))
    im[:, :, :3] = imdiff
    im[:, :, 3] = rg
    im[:, :, 4] = by

    # ipdb.set_trace()
    return im


def main():
    debug = True
    # debug = False
    trainset = glob(join("face_training", "face*.png"))
    si = ScreenImage()
    score = np.zeros(len(trainset))

    for i, train in enumerate(trainset):
        t0 = time()
        im_num = int(re.findall('\d+', train)[0])
        truth_name = glob(join("face_training_groundtruth",
                               "*mask%d.png" % im_num))[0]

        # print("Loading training and truth images %s... " % train)
        im_train = plt.imread(train)
        im_truth = plt.imread(truth_name)[:, :, 0].astype(np.uint8)
        (w, h, d) = im_train.shape

        # print("Extracting (r, g) [R, G, B] features... ")
        # im_train = gaussian_filter(im_train, 1.5)
        im_train = log_opponent(im_train)
        # fvec = extract_features(im_train)
        fvec = im_train.reshape((-1, 5)).astype('float32')

        # print("Fitting model to features ... ")
        kmeans = KMeans(n_clusters=8, tol=.001, n_jobs=4,
                        max_iter=100, n_init=10).fit(fvec)

        # print("Predicting color indices of the image... ")
        labels = kmeans.predict(fvec)
        im_lb = labels.reshape((w, h)).astype(np.uint8)

        # print("Getting majority labels for classifier... ")
        maj_lb, im_val, im_lbl = get_majority_label(kmeans, im_lb, im_truth)

        # print("Fitting Naive Bayes model... ")
        # clf = GaussianNB()

        score[i] = jaccard_similarity_score(im_truth, im_lbl)
        # print "label %d, Jaccard %.2f" % (maj_lb - 1, score[i])
        print "%s, done in %.2f sec, Score %.2f" % (train, time() - t0, score[i])

        if debug:
            si.show([im_train[:, :, :3], im_truth, im_lb * im_truth],
                    ["Training", "Truth Mask", "Similarity %.2f" % (score[i])])

    print score
    print "Average score: %.2f, K: %d" % (np.mean(score), kmeans.n_clusters)

if __name__ == '__main__':
    main()
