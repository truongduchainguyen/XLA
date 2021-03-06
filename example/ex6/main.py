from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('Demo_Ex6.ui', self)

        self.image = None

        self.btn_open: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_open')
        self.btn_show3D: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_show3D')
        self.btn_showImage: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_showImage')
        self.lbl_input_image: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_input_image')
        self.lbl_output_image: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_output_image')
        #self.lbl_kernel: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_kernel')
        #self.lbl_kernel_2: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_kernel_2')
        self.lbl_kernel_3: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_kernel_3')
        self.lbl_kernel_4: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_kernel_4')
        self.lbl_n: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_n')
        self.lbl_sigma: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_sigma')
        self.lbl_D0: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_D0')
        self.lbl_Params: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_Params')



        self.radioButton: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, 'radioButton')
        self.radioButton_2: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, 'radioButton_2')
        self.radioButton_3: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, 'radioButton _3')
        self.radioButton_4: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, 'radioButton_4')
        self.radioButton_5: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, 'radioButton_5')
        self.menubar: QtWidgets.QMenuBar = self.findChild(QtWidgets.QMenuBar, 'menubar')
        self.menuFile: QtWidgets.QMenu = self.findChild(QtWidgets.QMenu, 'menuFile')
        self.actionOpen: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionOpen')
        self.actionExit: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionExit')

        self.btn_open.clicked.connect(self.openFile)
        self.btn_open.clicked.connect(lambda: self.isClicked("btn_open"))



        #self.btn_open.clicked.connect(lambda: self.isClicked("btn_open"))
        #self.btn_open.clicked.connect(lambda: self.isClicked("btn_open"))
        self.show()

    def isClicked(self, obj):
        print("{} was clicked".format(obj))

    def openFile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', 'Image files (*.png *.xpm *.jpg)')
        if filename[0]:
            self.image = cv2.imread(filename[0])
            self.showImage(self.lbl_input_image, self.image)
            print(filename)
        else:
            print("invalid file")

    def showImage(self, label: QtWidgets.QLabel, cv_img):
        if cv_img is None:
            cv_img = self.image

        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QtGui.QPixmap(q_img))
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tmp = UI()
    app.exec_()
