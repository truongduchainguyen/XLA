from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('masterUI/master.ui', self)

        self.image = None

        self.lbl_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_input_img')
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
        self.btn_sobel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_sobel')
        self.btn_prewitt: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_prewitt')
        self.btn_suffer_1: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_suffer_1')
        self.btn_suffer_2: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_suffer_2')

        self.grb_Tool: QtWidgets.QGroupBox = self.findChild(QtWidgets.QGroupBox, 'grb_Tool')

        #self.menuFile = self.findChild(QtWidgets.)
        #self.menuABout = self.findChild(QtWidgets.)

        self.actionOpen: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionOpen')
        self.actionExit: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionExit')

        #connection
        self.btn_open.clicked.connect(self.openFile)
        self.btn_open.clicked.connect(lambda: self.isClicked("btn_open"))
        self.btn_apply.clicked.connect(lambda: self.isClicked("btn_apply"))
        self.btn_sobel.clicked.connect(lambda: self.isClicked("btn_sobel"))
        self.btn_sobel.clicked.connect(self.sobel)
        self.btn_prewitt.clicked.connect(lambda: self.isClicked("btn_prewitt"))
        self.btn_prewitt.clicked.connect(self.prewitt)
        self.btn_suffer_1.clicked.connect(lambda: self.isClicked("btn_suffer_1"))
        self.btn_suffer_1.clicked.connect(self.idontwanttosuffer1)
        self.btn_suffer_2.clicked.connect(lambda: self.isClicked("btn_suffer_2"))
        self.btn_suffer_2.clicked.connect(self.idontwanttosuffer2)


        self.show()

    def isClicked(self, obj):
        print("{} was clicked".format(obj))

    def openFile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', 'Image files (*.png *.xpm *.jpg)')
        if filename[0]:
            self.image = cv2.imread(filename[0])
            self.showImage(self.lbl_input_img, self.image)
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
    
    def sobel(self):
        hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        hy = np.array([[1, 2, -1], [0, 0, 0], [-1, -2, -1]])

        magnitude = np.abs(hx) + np.abs(hy)
        direct = np.arctan(hy / hx)

        #image = cv2.Sobel(self.image, -1, hx, hy)
        
        image1 = cv2.filter2D(self.image, -1, hx)
        cv2.imshow("sobel_hx", image1)
        image2 = cv2.filter2D(self.image, -1, hy)
        cv2.imshow("sobel_hy", image2)

        image3 = cv2.filter2D(self.image, -1, magnitude)
        cv2.imshow("sobel_mag", image3)
        image4 = cv2.filter2D(self.image, -1, direct)
        cv2.imshow("sobel_arctan", image4)

    def prewitt(self):
        hx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        hy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        magnitude = magnitude = np.abs(hx) + np.abs(hy)
        direct = np.arctan(hy / hx)

        image1 = cv2.filter2D(self.image, -1, hx)
        cv2.imshow("prewitt_hx", image1)
        image2 = cv2.filter2D(self.image, -1, hy)
        cv2.imshow("prewitt_hy", image2)
        image3 = cv2.filter2D(self.image, -1, magnitude)
        cv2.imshow("prewitt_mag", image3)
        image4 = cv2.filter2D(self.image, -1, direct)
        cv2.imshow("prewitt_arctan", image4)        
    
    def idontwanttosuffer1(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(self.image,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(self.image,127,255,cv2.THRESH_TOZERO)
        ret,thresh3 = cv2.threshold(self.image,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(self.image,127,255,cv2.THRESH_OTSU)

        titles = ['Original Image','BINARY','TOZERO','TRUNC','OTSU'] #
        images = [self.image, thresh1, thresh2, thresh3, thresh4] #
        for i in range(5):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
    
    def idontwanttosuffer2(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(self.image,5)
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for i in range(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tmp = UI()
    app.exec_()
