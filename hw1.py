#!/usr/bin/env python
import matplotlib
matplotlib.use("Qt4Agg")
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from os import path
print "OpenCV version " + cv2.__version__
print "matplotlib version " + matplotlib.__version__


def imshow_and_move(im, name, order):
    a = np.shape(im)
    cv2.imshow(name, im)
    cv2.moveWindow(name, (order % 3) * (a[0]), order // 3 * (a[1]))


def find_colorize_binary(im, fullpath, thresh, maxcontour):
    # parse savename
    filename, ext = path.splitext(fullpath)
    filename = path.split(filename)[-1]

    # Threshold
    val, imth = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite("images.out/" + filename + "_thresh" + ext, imth)
    # imth = cv2.adaptiveThreshold(im, 255,
    #                              cv2.ADAPTIVE_THRESH_MEAN_C,
    #                              cv2.THRESH_BINARY, 31, 2)

    # # Initial Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    im = cv2.morphologyEx(imth, cv2.MORPH_CLOSE, kernel)

    # Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    # Final Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("images.out/" + filename + "_morph" + ext, im)
    imorph = im

    # Find and Draw Contours
    _, contours, _ = cv2.findContours(im, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)

    im = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    for i in range(len(contours)):
        if len(contours[i]) < maxcontour:
            fillcolor = (0, 0, 0)
        else:
            fillcolor = tuple(np.random.randint(256, size=3))
        cv2.drawContours(im, contours, i, fillcolor, -1)

    cv2.imwrite("images.out/" + filename + "_final" + ext, im)
    return imth, imorph, im


def main():
    images = glob(path.join("images.in", "hw1_*.png"))
    maxcontour = 100
    thresh = [131, 133, 131]
    for idx, fullpath in enumerate(images):
        # Read file convert to gray
        im = cv2.cvtColor(cv2.imread(fullpath), cv2.COLOR_BGR2GRAY)
        imshow_and_move(im, "Original: " + fullpath, 0)
        im1, im2, im3 = find_colorize_binary(im, fullpath, thresh[idx], maxcontour)

        imshow_and_move(im1, "Threshold: thresh %d" % thresh[idx], 1)
        imshow_and_move(im2, "Morphology: " + fullpath, 2)
        imshow_and_move(im3, "Colored: " + fullpath, 3)

        # Closing
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
