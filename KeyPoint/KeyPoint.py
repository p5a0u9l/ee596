#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt4Agg")
from skimage.feature import corner_harris
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.io import ImageCollection
from skimage import img_as_ubyte
from os.path import splitext, split
from glob import glob
import numpy as np
import cv2
from tabulate import tabulate
import pickle


def interest_operators(im, prefix):
    w, h = im.shape
    # Harris Response
    H = corner_harris(im)
    im_harr = np.tile(im[:, :, np.newaxis], (1, 1, 3))
    im_harr[H > np.mean(H) + 0.8*np.std(H), 0] = 255

    # SURF features
    min_hessian = 4000
    surf = cv2.xfeatures2d.SURF_create(min_hessian)
    kp, des = surf.detectAndCompute(im, None)
    im_surf = np.zeros((w, h, 3), np.uint8)
    cv2.drawKeypoints(im, kp, im_surf,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(im, None)
    im_sift = img_as_ubyte(np.zeros((w, h, 3)))
    cv2.drawKeypoints(im, kp, im_sift,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Write Out Results
    imsave("im.out/" + prefix + "_Harris_Response.jpg", im_harr)
    imsave("im.out/" + prefix + "_SIFT_KeyPoints.jpg", im_sift)
    imsave("im.out/" + prefix + "_SURF_KeyPoints.jpg", im_surf)


def homographize(ic, operator=None):
    bf = cv2.BFMatcher()

    if operator == "SURF":
        min_hessian = 400
        ops = cv2.xfeatures2d.SURF_create()
    elif operator == "SIFT":
        ops = cv2.xfeatures2d.SIFT_create()

    kpt0, des0 = ops.detectAndCompute(ic[1], None)
    kpt1, des1 = ops.detectAndCompute(ic[0], None)

    matches = bf.knnMatch(des0, des1, k=2)
    mp0 = np.array([kpt0[match[0].queryIdx].pt for match in matches])
    mp1 = np.array([kpt1[match[0].trainIdx].pt for match in matches])
    H, _ = cv2.findHomography(mp0, mp1, cv2.RANSAC)

    if operator == "SURF":
        tablex = []
        tabley = []
        # import ipdb; ipdb.set_trace()

        for i in range(5):
            y = matches[i][0]
            if i % 2:
                x = kpt0[np.random.random_integers(1000)]
                fn = ic.files[0].replace("im.in/", "")
            else:
                x = kpt1[np.random.random_integers(1000)]
                fn = ic.files[1].replace("im.in/", "")
            if i == 0:
                headerx = [name for name in dir(x) if not name.startswith('__')]
                headery = [name for name in dir(y) if not name.startswith('__')]
            valsx = [getattr(x, name) for name in headerx]
            valsy = [getattr(y, name) for name in headery]
            valsx.insert(0, fn); valsy.insert(0, fn)
            tablex.append(valsx); tabley.append(valsy)

        with open(fn[0] + "_homog_cache", "w+") as file_:
            file_.write(tabulate(H, tablefmt="pipe"))

        if fn[0] == "a":
            pickle.dump(tablex, open("xtable_cache", "w"))
            pickle.dump(tabley, open("ytable_cache", "w"))
        else:
            with open("_SURF_Map_table", "w+") as file_:
                atable = pickle.load(open("ytable_cache", "r"))
                [tabley.append(val) for val in atable]
                headery.insert(0, "Image Name")
                file_.write(tabulate(tabley, headery, tablefmt="pipe"))
            with open("_SURF_KeyPoints_table", "w+") as file_:
                atable = pickle.load(open("xtable_cache", "r"))
                [tablex.append(val) for val in atable]
                headerx.insert(0, "Image Name")
                file_.write(tabulate(tablex, headerx, tablefmt="pipe"))
    return H


def warp_and_stitch(homog, prefix):
    im = [imread(fn) for fn in (glob("im.in/" + prefix + "*.jpg"))]
    (h, w, d) = im[0].shape
    warped = cv2.warpPerspective(im[0], homog, (2*w, h))
    warped[0:h, 0:w, :] = im[1]
    x = np.sum(np.sum(warped, axis=2), axis=0)
    edge_idx = np.where(x == 0)[0][0]
    return warped[:, :edge_idx, :]


def imread_wrapper(fname):
    # for hw3, reads an image, converts to gray as uint8
    return img_as_ubyte(rgb2gray(imread(fname)))


def main():
    ic = ImageCollection("im.in/*.jpg", load_func=imread_wrapper)
    for i, im in enumerate(ic):
        prefix = splitext(split(ic.files[i])[1])[0]
        print "%s: Interest Operators..." % (prefix)
        interest_operators(im, prefix)

    for op in ["SURF", "SIFT"]:
        for prefix in ["a"]:
            print "%s: %s: Matching Descriptors and Computing"\
                " Homography..." % (op, prefix)
            ic = ImageCollection("im.in/" + prefix + "*.jpg", load_func=imread_wrapper)
            H = homographize(ic, operator=op)
            print "%s: %s: Warp, Combine, Save..." % (op, prefix)
            stitched = warp_and_stitch(H, prefix)
            imsave("im.out/" + op + "_" + prefix + "_Combined.jpg", stitched)


if __name__ == '__main__':
    main()
