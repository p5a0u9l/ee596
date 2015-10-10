#!/usr/bin/env python
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.gridspec as gridspec


def imshow_and_move(im, name, order):
    a = np.shape(im)
    cv2.imshow(name, im)
    cv2.moveWindow(name, (order % 3) * (a[0]), order // 3 * (a[1]))

im = cv2.imread("hw1/kidney.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imshow_and_move(im, 'Original', 0)

# Threshold
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)

for i in range(5):
    g = cv2.GaussianBlur(im, (2 * i + 1, 2 * i + 1), 0)
    ax = fig.add_subplot(gs[i])
    imshow_and_move(g, 'Threshold', i)
    ax.hist(g.ravel(), 256)

plt.show()

mu, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
imshow_and_move(im, 'Threshold', 1)

# Closing
kernel = np.ones((2, 2), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
imshow_and_move(im, 'Morph: Close', 2)

# Opening
kernel = np.ones((8, 8), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
imshow_and_move(im, 'Morph: Open', 3)

# Contours
_, contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (255, 255, 255), 2)

# Plots
imshow_and_move(im, 'Contours', 4)

cv2.waitKey(0)
cv2.destroyAllWindows()
