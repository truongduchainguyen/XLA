from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
    img = cv2.imread("brain.jpg")
    img = cv2.resize(img, (700, 500))

    arr_gx = np.array([[0,0,0],[-1,2,-1],[0,0,0]])
    arr_gy = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])

    Gx = cv2.filter2D(img, -1, arr_gx)
    Gy = cv2.filter2D(img, -1, arr_gy)

    magnitude = abs(Gx) + abs(Gy)

    blur = cv2.blur(img, (100,100))
    sharpen = img - blur
    Theta = np.arctan(Gy/Gx)

    cv2.imshow("blur", blur)
    cv2.waitKey(0)

    cv2.imshow("sharpen", sharpen)
    cv2.waitKey(0)

    cv2.imshow("Gx", Gx)
    cv2.waitKey(0)

    cv2.imshow("Gy", Gy)
    cv2.waitKey(0)

    cv2.imshow("magnitude", magnitude)
    cv2.waitKey(0)

    cv2.imshow("Theta", Theta)
    cv2.waitKey(0)
