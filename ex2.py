# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ex2.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(675, 580)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbImage = QtWidgets.QLabel(self.centralwidget)
        self.lbImage.setEnabled(True)
        self.lbImage.setGeometry(QtCore.QRect(30, 10, 621, 421))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(36)
        self.lbImage.setFont(font)
        self.lbImage.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lbImage.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.lbImage.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.lbImage.setObjectName("lbImage")
        self.btnLinear = QtWidgets.QPushButton(self.centralwidget)
        self.btnLinear.setGeometry(QtCore.QRect(590, 510, 75, 23))
        self.btnLinear.setObjectName("btnLinear")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 470, 351, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 440, 261, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 500, 381, 41))
        self.label_3.setObjectName("label_3")
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
        self.action_Exit_2.triggered.connect(MainWindow.close)
        self.action_Open.triggered.connect(self.open_file)
        self.btnLinear.clicked.connect(self.push)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbImage.setText(_translate("MainWindow", "        Image Here"))
        self.btnLinear.setText(_translate("MainWindow", "Linear"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Nguyễn Ngọc Toàn 44.01.104.197</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Dương Tiến 44.01.104.193</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Trương Đức Hải Nguyên 44.01.104.155</span></p></body></html>"))
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
        des = filename[0]
        path = filename[0]
        pixmap = QPixmap(path)
        self.lbImage.setPixmap(pixmap)
        self.lbImage.setScaledContents(True)
        self.lbImage.show()
        return des

    def linear(self):
        img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg')
        imgMean = np.mean(img)
        imgStd = np.std(img)
        outMean = 100
        outStd = 20
        scale = outStd / imgStd
        shift = outMean - scale * imgMean
        imgLinear = shift + scale * img
        return imgLinear

    def push(self):
        from PIL import Image
        image = Image.fromarray(self.linear().astype(np.uint8))
        image.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
