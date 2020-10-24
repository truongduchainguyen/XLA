from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('master.ui', self)

        '''preloaded'''
        self.image = None
        self.path = ''

        '''Find Children'''
        #labels
        #self.lbl_: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_')
        self.lbl_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_input_img')
        self.lbl_zoom_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_zoom_input_img')
        #buttons
        #self.btn_: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_') #use this for template
        self.btn_apply: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_apply')
        self.btn_adaptive_threshold: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_adaptive_threshold')
        self.btn_brightness: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_brightness')
        self.btn_denoise: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_denoise')
        self.btn_filter: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_filter')
        self.btn_gamma: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_gamma')
        self.btn_grabcut: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_grabcut')
        self.btn_invert_color: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_invert_color')
        self.btn_noise: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_noise')
        self.btn_open: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_open')
        self.btn_prewitt: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_prewitt')
        self.btn_rotate: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_rotate')
        self.btn_roberts: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_roberts')
        self.btn_sobel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_sobel')
        self.btn_show_histogram: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_show_histogram')
        self.btn_show_diagram_3d: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_show_diagram_3d')
        self.btn_transform: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_transform')
        self.btn_threshold: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_threshold')

        #group boxes
        self.grb_Tool: QtWidgets.QGroupBox = self.findChild(QtWidgets.QGroupBox, 'grb_Tool')

        #self.menuFile = self.findChild(QtWidgets.)
        #self.menuABout = self.findChild(QtWidgets.)

        #actions
        self.actionOpen: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionOpen')
        self.actionExit: QtWidgets.QAction = self.findChild(QtWidgets.QAction, 'actionExit')
        '''end findChildren'''
        '''connection'''
        #is_clicked
        self.btn_apply.clicked.connect(lambda: self.isClicked('btn_apply'))
        self.btn_adaptive_threshold.clicked.connect(lambda: self.isClicked('btn_adaptive_threshold'))
        self.btn_brightness.clicked.connect(lambda: self.isClicked('btn_brightness'))
        self.btn_denoise.clicked.connect(lambda: self.isClicked('btn_denoise'))
        self.btn_filter.clicked.connect(lambda: self.isClicked('btn_filter'))
        self.btn_gamma.clicked.connect(lambda: self.isClicked('btn_gamma'))
        self.btn_grabcut.clicked.connect(lambda: self.isClicked('btn_grabcut'))
        self.btn_invert_color.clicked.connect(lambda: self.isClicked('btn_invert_color'))
        self.btn_noise.clicked.connect(lambda: self.isClicked('btn_noise'))
        self.btn_open.clicked.connect(lambda: self.isClicked('btn_open'))
        self.btn_prewitt.clicked.connect(lambda: self.isClicked('btn_prewitt'))
        self.btn_rotate.clicked.connect(lambda: self.isClicked('btn_rotate'))
        self.btn_roberts.clicked.connect(lambda: self.isClicked('btn_roberts'))
        self.btn_sobel.clicked.connect(lambda: self.isClicked('btn_sobel'))
        self.btn_show_histogram.clicked.connect(lambda: self.isClicked('btn_show_histogram'))
        self.btn_show_diagram_3d.clicked.connect(lambda: self.isClicked('btn_show_diagram_3d'))
        self.btn_transform.clicked.connect(lambda: self.isClicked('btn_transform'))
        self.btn_threshold.clicked.connect(lambda: self.isClicked('btn_threshold'))

        #buttons
        self.btn_open.clicked.connect(self.openFile)
        self.btn_sobel.clicked.connect(self.sobel)
        self.btn_prewitt.clicked.connect(self.prewitt)
        self.btn_threshold.clicked.connect(self.applyThreshold)
        self.btn_adaptive_threshold.clicked.connect(self.applyAdaptiveThreshold)
        self.btn_grabcut.clicked.connect(self.grabcut)
        #actions
        self.actionOpen.triggered.connect(self.openFile)
        '''end connection'''

        self.show()

    def isClicked(self, obj):
        print("{} was clicked".format(obj))

    def createNamedWindow(self, name, src):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, src)


    def openFile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', 'Image files (*.png *.xpm *.jpg *.tif)')
        #print(filename)
        if filename[0] != '' and filename[0] != None:
            self.path = filename[0]
            self.image = cv2.imread(filename[0])
            self.showImage(self.lbl_input_img, self.image)
        else:
            print("invalid file")

    def showImage(self, label: QtWidgets.QLabel, cv_img):
        if cv_img is None:
            cv_img = self.image
        if self.image is not None:
                height, width = cv_img.shape[:2]
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
                label.setPixmap(QtGui.QPixmap(q_img))
        else:
            print("Warning: self.image is empty.")
    
    def sobel(self):
        hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        hy = np.array([[1, 2, -1], [0, 0, 0], [-1, -2, -1]])

        magnitude = np.abs(hx) + np.abs(hy)
        direct = np.arctan(hy / hx)

        #image = cv2.Sobel(self.image, -1, hx, hy)

        if self.image is not None:
            image1 = cv2.filter2D(self.image, -1, hx)
            image2 = cv2.filter2D(self.image, -1, hy)
            image3 = image1 + image2

            self.showImage(self.lbl_input_img, image3)
            #self.createNamedWindow("sobel_hx", image1)
            #self.createNamedWindow("sobel_hy", image2)
        else:
            print("Warning: self.image is empty.")

    def prewitt(self):
        hx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        hy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        #magnitude and direction
        magnitude = np.abs(hx) + np.abs(hy)
        direct = np.arctan(hy / hx)

        if self.image is not None:
            image1 = cv2.filter2D(self.image, -1, hx)
            image2 = cv2.filter2D(self.image, -1, hy)
            image3 = image1 + image2

            self.showImage(self.lbl_input_img, image3)

            #self.createNamedWindow("prewitt_hx", image1)
            #self.createNamedWindow("prewitt_hy", image2)
            #self.createNamedWindow("prewitt", image3)

        else:
            print("Warning: self.image is empty.")
    
    def applyThreshold(self):
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
            ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
            ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_OTSU)

            titles = ['Original Image','BINARY','TOZERO','TRUNC','OTSU'] #
            images = [image, thresh1, thresh2, thresh3, thresh4] #
            for i in range(5):
                plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            plt.show()
        else:
            print("Warning: self.image is empty.")

    def applyAdaptiveThreshold(self):
        # this function kill the program
        # do not run

        if self.image is not None:
            result = self.apply_filter()

            output = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Warning: self.image is empty.")

    def grabcut(self): #grabcut
        if self.image is not None:
            img = self.image
            mask = np.zeros(img.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            rect = (50,50,450,290)
            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img = img*mask2[:,:,np.newaxis]
            plt.imshow(img),plt.colorbar(),plt.show()
        else:
            print("Warning: self.image is empty.")

    # def idontwanttosuffer4(self):
    def low_gauss(self, sigma = 3):
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sx, sy = image.shape
            x = np.arange(-sy / 2, sy / 2)
            y = np.arange(-sx / 2, sx / 2)
            [x, y] = np.meshgrid(x, y)
            mg = np.sqrt(x ** 2 + y ** 2)
            H = np.exp((-mg) / 2 * (sigma ** 2))
            # print(H)
            return H

    def apply_filter(self):
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            H = self.low_gauss(0.4)
            G = np.fft.fftshift(np.fft.fft2(image))
            Ip = G * H
            im = np.abs(np.fft.ifft2(np.fft.fftshift(Ip)))
            return np.uint8(im)
        else:
            print("Warning: self.image is empty")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tmp = UI()
    app.exec_()
