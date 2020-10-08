from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('master.ui', self)

        self.image = None

        self.lbl_output_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_output_img')
        self.lbl_zoom_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_zoom_input_img')

        self.btn_open: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_open')
        self.btn_apply: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_apply')
        self.btn_filter: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_filter')
        #self.btn_ = self.findChild(QtWidgets.QPushButton, '')
        self.btn_rotate: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_rotate')
        self.btn_brightness: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_brightness')
        self.btn_gamma: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_gamma')
        self.btn_invert_color: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_invert_color')
        self.btn_show_histogram: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_show_histogram')
        self.btn_show_diagram_3d: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_show_diagram_3d')
        self.btn_denoise: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_denoise')
        self.btn_noise: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_noise')
        self.btn_transform: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_transform')

        self.grb_Tool: QtWidgets.QGroupBox = self.findChild(QtWidgets.QGroupBox, 'grb_Tool')

        #self.menuFile = self.findChild(QtWidgets.)
        #self.menuABout = self.findChild(QtWidgets.)

        self.actionOpen: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionOpen')
        self.actionExit: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionExit')

        #connection
        self.btn_open.clicked.connect(self.openFile)
        self.btn_open.clicked.connect(lambda: self.isClicked("btn_open"))
        self.btn_apply.clicked.connect(lambda: self.isClicked("btn_apply"))

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
