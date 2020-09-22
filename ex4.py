# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ex2.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtGui import QPixmap
from matplotlib import pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(675, 585)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 470, 351, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 440, 261, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 500, 381, 41))
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setEnabled(True)
        self.spinBox.setGeometry(QtCore.QRect(420, 510, 42, 22))
        self.spinBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setMinimum(1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_2.setGeometry(QtCore.QRect(480, 510, 42, 22))
        self.spinBox_2.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spinBox_2.setMinimum(1)
        self.spinBox_2.setObjectName("spinBox_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 420, 131, 41))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(400, 480, 131, 41))
        self.label_5.setObjectName("label_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(570, 510, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(36, 42, 591, 361))
        self.label_6.setObjectName("label_6")
        self.spinBox_3 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_3.setGeometry(QtCore.QRect(560, 430, 42, 22))
        self.spinBox_3.setObjectName("spinBox_3")
        self.spinBox_3.setMinimum(1)
        self.spinBox_3.setMaximum(9)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(510, 390, 141, 41))
        self.label_7.setObjectName("label_7")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(380, 460, 219, 19))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.rdbMean = QtWidgets.QRadioButton(self.widget)
        self.rdbMean.setObjectName("rdbMean")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.rdbMean)
        self.horizontalLayout.addWidget(self.rdbMean)
        self.rdb_Blur = QtWidgets.QRadioButton(self.widget)
        self.rdb_Blur.setObjectName("rdb_Blur")
        self.buttonGroup.addButton(self.rdb_Blur)
        self.horizontalLayout.addWidget(self.rdb_Blur)
        self.rdb_Gauss = QtWidgets.QRadioButton(self.widget)
        self.rdb_Gauss.setObjectName("rdb_Gauss")
        self.buttonGroup.addButton(self.rdb_Gauss)
        self.horizontalLayout.addWidget(self.rdb_Gauss)
        self.rdb_Median = QtWidgets.QRadioButton(self.widget)
        self.rdb_Median.setObjectName("rdb_Median")
        self.buttonGroup.addButton(self.rdb_Median)
        self.horizontalLayout.addWidget(self.rdb_Median)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 675, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setCheckable(True)
        self.actionExit.setObjectName("actionExit")
        self.actionOpen_2 = QtWidgets.QAction(MainWindow)
        self.actionOpen_2.setObjectName("actionOpen_2")
        self.actionExit_2 = QtWidgets.QAction(MainWindow)
        self.actionExit_2.setCheckable(True)
        self.actionExit_2.setObjectName("actionExit_2")
        self.actionOpen_3 = QtWidgets.QAction(MainWindow)
        self.actionOpen_3.setObjectName("actionOpen_3")
        self.action_Exit = QtWidgets.QAction(MainWindow)
        self.action_Exit.setObjectName("action_Exit")
        self.action_Open = QtWidgets.QAction(MainWindow)
        self.action_Open.setObjectName("action_Open")
        self.action_Exit_2 = QtWidgets.QAction(MainWindow)
        self.action_Exit_2.setCheckable(False)
        self.action_Exit_2.setIconVisibleInMenu(False)
        self.action_Exit_2.setObjectName("action_Exit_2")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_Open)
        self.menuFile.addAction(self.action_Exit_2)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.action_Open.triggered.connect(self.open_file)
        self.action_Exit_2.triggered.connect(MainWindow.close)
        self.pushButton.clicked.connect(self.show)
        self.action_Exit_2.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Nguyễn Ngọc Toàn 44.01.104.197</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Dương Tiến 44.01.104.193</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Trương Đức Hải Nguyên 44.01.104.155</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Type of filter:</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Size of filter:</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Apply"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:22pt; font-weight:600;\">     Image here</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-style:italic;\">Kernel size for Median:</span></p></body></html>"))
        self.rdbMean.setText(_translate("MainWindow", "Mean"))
        self.rdb_Blur.setText(_translate("MainWindow", "Blur"))
        self.rdb_Gauss.setText(_translate("MainWindow", "Gauss"))
        self.rdb_Median.setText(_translate("MainWindow", "Median"))
        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionOpen_2.setText(_translate("MainWindow", "Open"))
        self.actionExit_2.setText(_translate("MainWindow", "Exit"))
        self.actionOpen_3.setText(_translate("MainWindow", "&Open"))
        self.action_Exit.setText(_translate("MainWindow", "&Exit"))
        self.action_Open.setText(_translate("MainWindow", "&Open"))
        self.action_Exit_2.setText(_translate("MainWindow", "&Exit"))



    def open_file(self):
        filename = QtWidgets.QFileDialog.getOpenFileName()
        path = filename[0]
        pixmap = QPixmap(path)
        self.image = cv2.imread(path)
        self.label_6.setPixmap(pixmap)
        self.label_6.setScaledContents(True)
        self.label_6.show()

    def Mean(self):
        img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg').astype(np.float32) / 255
        blur = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg').astype(np.float32) / 255
        blur -= img.mean()
        blur /= img.std()
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Mean')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey(0)
        print('Mean')

    def Blur(self):
        img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg')
        blur = cv2.blur(img, (self.spinBox.value(), self.spinBox_2.value()))
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()
        print('Blur')

    def Gauss(self):
        img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg')
        blur = cv2.GaussianBlur(img, (self.spinBox.value(), self.spinBox_2.value()), 0)
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Gauss')
        plt.xticks([]), plt.yticks([])
        plt.show()
        print('Gauss')

    def Median(self):
        img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/mona_medianfilter.jpg')
        blur = cv2.medianBlur(img, self.spinBox_3.value())
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Median')
        plt.xticks([]), plt.yticks([])
        plt.show()
        print('Median')
    def show(self):
        if(self.rdbMean.isChecked()):
            # print('Mean')
            # print('Apply is clicked')
            self.Mean()
        if(self.rdb_Blur.isChecked()):
            self.Blur()
        if(self.rdb_Gauss.isChecked()):
            self.Gauss()
        if(self.rdb_Median.isChecked()):
            self.Median()
        else:
            self.label_6.setTextFormat("The filter hasn't been selected yet!!")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
