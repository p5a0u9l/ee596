__author__ = 'adamspr'

import cv2
from glob import glob
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt

def display_images(base_path, ext=None):
    if ext == None: ext = ".png"
    imFiles = glob(join(base_path, "*" + ext))
    for imF in imFiles:
        im = cv2.imread(imF, 0)
        cv2.imshow(imF, im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imshow_and_move(im, name, order):
    a = np.shape(im)
    cv2.imshow(name, im)
    cv2.moveWindow(name, (order % 3)*(a[0]+10), order//3*(a[1] + 40))

def main():
    # display_images("hw1")

    # reference
    im = cv2.imread("hw1/kidney-regions-sm.png")
    imshow_and_move(im, "Organs: Reference", 0)

    # original
    im = cv2.imread("hw1/kidney.png")
    imshow_and_move(im, "Organs: Original", 1)

    # histogram
    if 0:
        x = im[:, :, 1]
        x = x.reshape(512**2)
        x_mu = x.astype('float')
        x_mu[np.where(x == 0)] = np.nan # prevent zeros biasing mean estimate
        x_mu = np.mean(x_mu)
        plt.subplot()
        plt.hist(x, 2**8)
        plt.title("Mean is %.3f" % x_mu)

    # thresholding
    _, im_thresh = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY_INV)
    imshow_and_move(im_thresh, "Organs: Thresholded", 2)

    # morphology - dilated
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    im_dilated = cv2.dilate(im_thresh, kernel)
    imshow_and_move(im_dilated, "Organs: Dilated", 3)

    # morphology - eroded
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    im_eroded = cv2.erode(im_dilated, kernel)
    imshow_and_move(im_eroded, "Organs: Eroded", 4)

    # get rid of images
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.close("all")

if __name__ == '__main__':
    main()
