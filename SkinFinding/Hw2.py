#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Paul Adams"
__assignment__ = "Homework 2"
__course__ = "EE596"
import matplotlib
matplotlib.use("Qt4Agg")
import pprint
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from glob import glob
from os.path import join
import numpy as np
from skimage.filters import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_similarity_score
from scipy.ndimage.filters import median_filter
from time import time
import sys
import ipdb
sys.path.append('/home/adamspr/ee596/MachineVision')
from ScreenImage import ScreenImage
from HW2_functions import cache_results, print_, get_groundname
verbosity = True


def get_norm_rg(im):
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


def get_log_opponent(im):
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
    intensity = im
    intensity[:, :, 0] = np.abs(im[:, :, 0] - i)
    intensity[:, :, 1] = np.abs(im[:, :, 1] - i)
    intensity[:, :, 2] = np.abs(im[:, :, 2] - i)
    im = np.zeros((im.shape[0], im.shape[1], 5))
    im[:, :, :3] = intensity
    im[:, :, 3] = rg
    im[:, :, 4] = by
    return im.reshape((-1, 5))


def get_truth_overlap(kmeans, im_rgb, mask, thresh=0):
    w, h = mask.shape
    K = kmeans.n_clusters
    labels = np.unique(mask)[1:]
    n_mask = np.count_nonzero(mask)
    # D is the output list of dicts
    D = [{} for l in labels]
    # Count occurences of each label in mask
    mask_counts = [np.count_nonzero(mask == lb) for lb in labels]
    for i, mask_count in enumerate(mask_counts):
        D[i]["Label"] = labels[i]
        D[i]["Overlap"] = mask_count / float(np.count_nonzero(im_rgb == labels[i]))
        D[i]["Count"] = mask_count
        D[i]["Center"] = kmeans.cluster_centers_[labels[i] - 1, :]
        if D[i]["Overlap"] > thresh:
            D[i]["Class"] = +1
        else:
            D[i]["Class"] = -1
    return D


def im2feature(im_name, params):
    im_train = imread(im_name)
    (w, h, d) = im_train.shape

    print_(verbosity, "\tExtracting feature vectors... ")
    if params["feature"] == "RGBrg":
        fvec_ = get_norm_rg(im_train)
    elif params["feature"] == "LogOp":
        fvec_ = get_log_opponent(im_train)
    elif params["feature"] == "RGB":
        fvec_ = get_norm_rg(im_train)[:, :3]
    elif params["feature"] == "rg":
        fvec_ = get_norm_rg(im_train)[:, 3:]
    elif params["feature"] == "BothRGBLOG":
        fvec_ = get_norm_rg(im_train)
        fvec_ = np.hstack([fvec_, get_log_opponent(im_train)])

    print_(verbosity, "\tClassifying features ...")
    kmeans = KMeans(n_clusters=params["n_cluster"], tol=.001, n_jobs=4,
                    max_iter=100, n_init=10).fit(fvec_)
    labels = kmeans.predict(fvec_) + 1   # offset by 10
    fvec = np.zeros(fvec_.shape)

    for i, lab in enumerate(labels):
        fvec[i, :] = kmeans.cluster_centers_[lab - 1, :]
    labels = labels.reshape((w, h)).astype(np.uint8)

    return labels, kmeans, fvec


def get_training_samples(trainset, params):
    Samples = np.zeros((200, len(params["feature"])))
    Labels = np.zeros(200,)
    k = 0
    for i, trainname in enumerate(trainset):
        print_(verbosity, "\tBeginning training and truth image set %d of %d... "
               % (i+1, len(trainset)))
        truthname = get_groundname(trainname)
        im_truth = imread(truthname)[:, :, 0].astype(np.uint8)
        rgb_lab, kmeans, fvec = im2feature(trainname, params)
        mask = rgb_lab * im_truth
        overlap = get_truth_overlap(kmeans, rgb_lab, mask,
                                    thresh=params["thresh"])
        print_(verbosity, "\tCache Samples/Labels ...\n")
        for lap in overlap:
            Samples[k, :] = lap["Center"]
            Labels[k] = lap["Class"]
            k += 1

    # Remove Missing Labels
    keepers = Labels != 0
    return Labels[keepers], Samples[keepers, :]


def paramterator():
    # Iterate over a range of features x overlap thresh x n_clusters to
    # find the best combo for each test image
    params = {}
    for clf in ["NB", "RF"]:
        for feature in ["RGB", "rg", "RGBrg", "LogOp", "BothRGBLOG"]:
            for thresh in [0.4, 0.5, 0.55, 0.6, 0.65]:
                for n_cluster in [4, 6, 8]:
                    params["classifier"] = clf
                    params["feature"] = feature
                    params["thresh"] = thresh
                    params["n_cluster"] = n_cluster
                    main(params)


def main(params):
    # Initialization
    si = ScreenImage()
    debug = False
    trainset = glob(join("face_training", "face*.png"))
    testset = glob(join("face_testing", "face*.png"))
    t0 = time()

    print_(verbosity, "Begin collecting training Samples")
    Labels, Samples = get_training_samples(trainset, params)
    print_(verbosity, "Success. Elapsed: %.2f s." % (time() - t0))

    print_(verbosity, "Begin classifier training using %s..."
           % (params["classifier"]))
    if params["classifier"] == "NB":
        clf = GaussianNB()
    elif params["classifier"] == "RF":
        clf = RandomForestClassifier()
    clf.fit(Samples, Labels)

    print_(verbosity, "Begin classifier prediction...")
    score = np.zeros(len(testset),)
    for i, testname in enumerate(testset):
        truthname = get_groundname(testname)
        im_lab, kmeans, fvec = im2feature(testname, params)
        im_skin = clf.predict(fvec).reshape(im_lab.shape)
        im_truth = imread(truthname)[:, :, 0].astype(np.uint8)
        im_skin = ((im_skin + 1)/2).astype(np.uint8)
        # ipdb.set_trace()
        score[i] = jaccard_similarity_score(im_truth, im_skin)
        if debug:
            si.show([imread(testname), im_truth, im_skin],
                    ["Test", "Truth", "Naive-Bayes: Score %.2f" % score[i]])
        print_(verbosity, "\tTest %d of %d, Score %.2f\n" % (i+1, len(testset), score[i]))
    print_(verbosity, "Success. Elapsed: %.2f s. Avg. Score %.2f" % (time() - t0, np.mean(score)))

    print_(verbosity, "Cache Results...")
    R = cache_results(score, params)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(R[-1:])


if __name__ == '__main__':
    if 0:
        params = {}
        params["classifier"] = "NB"
        params["feature"] = "rg"
        params["thresh"] = 0.55
        params["n_cluster"] = 8
        main(params)
    else:
        paramterator()
