# importing required libraries of opencv
import cv2
import  numpy as np

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.resize(cv2.imread('data/cats.jpg',0),(700,500))
equ = cv2.equalizeHist(img)
# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
# show the plotting graph of an image
plt.plot(histr)
plt.show()
import sys
from PyQt5 import QtWidgets, QtCore, uic
import cv2
import numpy as np

def changebrighness(img, val):
    return img + val

def showimg(img):
    cv2.imshow('show_img', img)
    cv2.waitKey(0)

def changegamma(img, val):
    return np.array(255*(img/255)**val, dtype=np.uint8)



app = QtWidgets.QApplication(sys.argv)
img = cv2.resize(cv2.imread('testimg.jpg'), (800, 600))
img1 = cv2.resize(cv2.imread('testimg2.jpg'), (800,600))

brightness_img = changebrighness(img, 10)
rotate_img =  cv2.rotate(img, cv2.ROTATE_180)
scale_img = cv2.resize(img, (200,100))
#apply log
log_img = (np.log(img+1)/(np.log(1+np.max(img)))) * 255
log_img = np.array(log_img, dtype=np.uint8)
#gamma
gamma_img = changegamma(img, -10)
#Add 2 imgs

add_img = cv2.add(img, img1)
addweight_img = cv2.addWeighted(img, 0.7, img1, 0.3, 0)


#showimg(img)
#showimg(img1)
#showimg(brightness_img)
#showimg(rotate_img)
#showimg(scale_img)
#showimg(log_img)
showimg(gamma_img)
#showimg(addweight_img)



#cv2.calcHist()
#histogram = cv2.calcHist([img], [0, 1], None, [180, 260])




