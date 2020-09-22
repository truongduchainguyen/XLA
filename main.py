from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMainWindow, QTextEdit, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
import cv2
import sys

class UI(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("BasicUI.ui", self)

        self.btnOpen = self.findChild(QWidget, "btnOpen")
        self.picBox = self.findChild(QWidget, "picBox")
        self.btnClose = self.findChild(QPushButton, "btnClose")
        self.urlBox = self.findChild(QLabel, "urlBox")

        self.btnOpen.clicked.connect(self.load_image)
        self.btnClose.clicked.connect(QApplication.instance().quit)

        self.show()       
        
    def load_image(self):
        filename = QtWidgets.QFileDialog.getOpenFileName()
        path = filename[0]
        pixmap = QtGui.QPixmap('path')
        #self.picBox.setPixmap(QtGui.QPixmap("path"))
        self.image = cv2.resize(cv2.imread(path, 0), (801, 391))
        #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        #self.picBox.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.image = cv2.imshow("lmao", self.image)
        #self.picBox.show()
        cv2.waitKey(0)
    
    def changebrighness(img, val):
        return img + val
    
    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tmp = UI()
    app.exec_()


