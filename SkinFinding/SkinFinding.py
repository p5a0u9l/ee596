#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Paul Adams"
__assignment__ = "Homework 2"
__course__ = "EE596"
import pprint
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from glob import glob
import pickle
from os.path import join
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_similarity_score
from skimage.filters import gaussian_filter
from skimage.color import rgb2gray
from scipy.ndimage.filters import median_filter
from time import time
import sys
import ipdb
sys.path.append('/home/adamspr/ee596/MachineVision')
from ScreenImage import ScreenImage
from HW2_functions import cache_results, print_, get_groundname
verbosity = False



def get_norm_rg(im):
    # Apply filtering
    im = median_filter(im, 2)
    # ipdb.set_trace()
    # image, int --> array, float
    rgb = im.reshape((-1, 3)).astype('float32')
    # sum along columns
    rgbsum = np.sum(rgb, axis=1)
    # prepare for array d3ivision
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
    rg = median_filter(rg, 2)
    by = median_filter(by, 2)
    i = median_filter(i, 2)
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
    labels = np.unique(mask)
    # D is the output list of dicts
    D = [{} for l in labels]
    # Count occurences of each label in mask
    mask_counts = [np.count_nonzero(mask == lb) for lb in labels]
    total_counts = [np.count_nonzero(im_rgb == lb) for lb in labels]
    for i, mask_count in enumerate(mask_counts):
        if total_counts[i] == 0:
            overlap = 0
        else:
            overlap = mask_count / float(total_counts[i])
        D[i]["Center"] = kmeans.cluster_centers_[labels[i], :]
        D[i]["Class"] = (overlap > thresh)*255
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
                    max_iter=300, n_init=52, verbose=0).fit(fvec_)
    labels = kmeans.predict(fvec_)
    fvec = np.zeros(fvec_.shape)

    for i, lab in enumerate(labels):
        fvec[i, :] = kmeans.cluster_centers_[lab, :]
    labels = labels.reshape((w, h)).astype(np.uint8)

    return labels, kmeans, fvec


def get_training_samples(trainset, params):
    Samples = np.zeros((200, len(params["feature"])))
    Labels = np.ones(200,)
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
    keepers = Labels != 1
    return Labels[keepers], Samples[keepers, :]


def main(params, train):
    si = ScreenImage()
    if train:
        # Initialization
        trainset = glob(join("face_training", "face*.png"))
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
        pickle.dump([clf, params], open(params["name"], "w"))
    else:
        testset = glob(join("face_testing", "face*.png"))
        print_(verbosity, "Begin classifier prediction...")
        score = np.zeros(len(testset),)
        models = glob("._*")

        for i, testname in enumerate(testset):
            im_orig = imread(testname)
            truthname = get_groundname(testname)
            im_skin = [[] for k in models]
            title = ["" for k in models]
            for j, model in enumerate(models):
                im_truth = rgb2gray(imread(truthname)).astype(np.uint8)*255
                pkl = pickle.load(open(model, "r"))
                clf = pkl[0]
                params = pkl[1]
                _, _, fvec = im2feature(testname, params)
                im_skin[j] = clf.predict(fvec).reshape(im_truth.shape).astype(np.uint8)
                score = jaccard_similarity_score(im_truth, im_skin[j], normalize=True)
                title[j] = "%s\nClassifier: %s, Thresh: %.2f\nK: %d, Score: %.2f" \
                    % (params["classifier"], params["feature"], params["thresh"],
                       params["n_cluster"], score)
                print_(verbosity, "\tTest %d of %d, Score %.2f\n" % (i+1, len(testset), score))

            si.show(testname, [im_orig, im_skin[0], im_skin[1],
                               im_skin[2], im_skin[3], im_skin[4]],
                    ["Original\n%s" % testname, title[0], title[1],
                    title[2], title[3], title[4]])

            # print("Success. Elapsed: %.2f s. Avg. Score %.2f" % (time() - t0, np.mean(score)))


def paramterator():
    # Iterate over a range of features x overlap thresh x n_clusters to
    # find the best combo for each test image
    params = {}
    for train in [False]:
        for thresh in [0.5]:
            for n_cluster in [8]:
                for clf in ["NB", "RF"]:
                    for feature in ["RGB", "rg", "LogOp"]:
                        params["classifier"] = clf
                        params["feature"] = feature
                        params["thresh"] = thresh
                        params["n_cluster"] = n_cluster
                        params["name"] = "._" + clf + "_" + feature + ".pkl"
                        main(params, train)

if __name__ == '__main__':
    paramterator()
