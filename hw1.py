__author__ = 'adamspr'


import cv2

im = cv2.imread("./hw1/e030.png", 0)
cv2.imshow('Test', im)
cv2.waitKey(0)
cv2.destroyAllWindows()