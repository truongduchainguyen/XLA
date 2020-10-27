import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import cm
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import random

from matplotlib.ticker import LinearLocator, FormatStrFormatter

from masterUI.dialog_kernel_size import Ui_Dialog_kernel_size
from masterUI.qdialog_brightness_contrast import Ui_dialog_brightness_contrast

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('masterUI/ui/master.ui', self)
        #uic.loadUi(r'D:/XLA/masterUI/ui/master.ui', self) #của tôi bị đéo load được UI nên phải bỏ như này,
        #có gì các gs cmt lại khi debug nhé

        '''preloaded'''
        self.original_image = None
        self.image = None
        self.path = ''

        '''for debugging'''
        #please delete this in the final project
        self.path = 'resource/Ididntseeshit.png'
        self.original_image = cv2.imread(self.path)
        self.image = cv2.imread(self.path)
        '''end'''

        '''Find Children'''
        #labels
        #self.lbl_: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_')
        self.lbl_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_input_img')
        self.lbl_zoom_input_img: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_zoom_input_img')
        #buttons
        #self.btn_: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_') #use this for template
        # self.btn_apply: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_apply')
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
        self.btn_kmeans: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_kmeans')
        self.btn_revert: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, 'btn_revert')
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
        # self.btn_apply.clicked.connect(lambda: self.isClicked('btn_apply'))
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
        self.btn_kmeans.clicked.connect(lambda: self.isClicked('btn_kmeans'))
        self.btn_revert.clicked.connect(lambda: self.isClicked('btn_revert'))
        self.btn_sobel.clicked.connect(lambda: self.isClicked('btn_sobel'))
        self.btn_show_histogram.clicked.connect(lambda: self.isClicked('btn_show_histogram'))
        self.btn_show_diagram_3d.clicked.connect(lambda: self.isClicked('btn_show_diagram_3d'))
        self.btn_transform.clicked.connect(lambda: self.isClicked('btn_transform'))
        self.btn_threshold.clicked.connect(lambda: self.isClicked('btn_threshold'))
        #buttons
        # self.btn_apply.clicked.connect(self.test_combobox)
        self.btn_adaptive_threshold.clicked.connect(self.applyAdaptiveThreshold)
        self.btn_brightness.clicked.connect(self.brightness)
        self.btn_filter.clicked.connect(self.filter)
        self.btn_grabcut.clicked.connect(self.grabcut)
        self.btn_open.clicked.connect(self.openFile)
        self.btn_invert_color.clicked.connect(self.invertColor)
        self.btn_prewitt.clicked.connect(self.prewitt)
        self.btn_revert.clicked.connect(self.revertToOriginal)
        self.btn_rotate.clicked.connect(self.rotateImage)
        self.btn_sobel.clicked.connect(self.sobel)
        self.btn_threshold.clicked.connect(self.applyThreshold)
        self.btn_show_diagram_3d.clicked.connect(self.draw3D)
        self.btn_show_histogram.clicked.connect(self.drawHistogram)
        self.btn_denoise.clicked.connect(self.denoise)
        self.btn_kmeans.clicked.connect(self.kmeans)

        #actions
        self.actionOpen.triggered.connect(self.openFile)
        '''end connection'''

        self.show()

    def isClicked(self, obj: str):
        '''
        Check if the object is clicked, result is printed in the console
        :param obj: the name of the object, in string
        :return: None
        '''
        print("{} was clicked".format(obj))

    def createNamedWindow(self, name: str, src: np.array):
        '''
        Show a popup window of the image
        :param name: The name of the namedwindow
        :param src: The image source in matrix or array
        :return: None
        '''
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, src)


    def openFile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', 'Image files (*.png *.xpm *.jpg *.tif)')
        #print(filename)
        if filename[0] != '' and filename[0] != None:
            self.path = filename[0]
            self.original_image = cv2.imread(filename[0])
            self.image = self.original_image
            self.showImage(self.lbl_input_img, self.image)
        else:
            print("invalid file")

    def showImage(self, label: QtWidgets.QLabel, cv_img = None):
        if cv_img is None:
            cv_img = self.image
        if cv_img is not None:
            height, width = cv_img.shape[:2]
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
            label.setPixmap(QtGui.QPixmap(q_img))
        else:
            print("Warning: self.image is empty.")

    def revertToOriginal(self):
        '''
        Revert back to the original loaded image
        :return: None
        '''
        if self.original_image is not None:
            self.image = self.original_image
            self.showImage(self.lbl_input_img)
        else:
            print('Warning: nothing was loaded, self.original_image is empty.')

    def rotateImage(self):
        if self.image is not None:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.showImage(self.lbl_input_img, self.image)
        else:
            print('Warning: nothing was loaded, self.original_image is empty.')

    def invertColor(self):
        '''
        Invert the color of the image, this is done by assigning: pixel_value = 255 - pixel_value
        :return: None
        '''
        #if cv_img is None:
        #    cv_img = self.image
        if self.image is not None:
            self.image = 255 - self.image
            self.showImage(self.lbl_input_img)
        else:
            print("Warning: self.image is empty.")

    def filter(self):
        if self.image is not None:
            dl_kernel_size = Ui_Dialog_kernel_size()
            dl_kernel_size.exec_()

            if dl_kernel_size.result() == QtWidgets.QDialog.Accepted:
                spb_x = dl_kernel_size.spb_x.value()
                spb_y = dl_kernel_size.spb_y.value()
                spb_z = dl_kernel_size.spb_z.value()
                filter_type = dl_kernel_size.cbox_blur.currentIndex()
                if filter_type == 0:
                    self.Mean()
                if filter_type == 1:
                    self.Blur(spb_x, spb_y)
                if filter_type == 2:
                    self.Median(spb_z)
                if filter_type == 3:
                    self.Gauss(spb_x, spb_y)

    def brightness(self):
        if self.image is not None:
            dl_brightness_contrast = Ui_dialog_brightness_contrast()
            dl_brightness_contrast.exec_()
            alpha = 1
            beta = 0
            if dl_brightness_contrast.result() == QtWidgets.QDialog.Accepted:
                alpha = dl_brightness_contrast.hslider_contrast.value()/10
                beta = dl_brightness_contrast.hslider_brightness.value()
                self.image = cv2.convertScaleAbs(self.image, alpha=alpha ,beta=beta)
                self.showImage(self.lbl_input_img, self.image)
            if dl_brightness_contrast.result() == QtWidgets.QDialog.Rejected:
                print('dialog rejected')
            #self.showImage(self.lbl_input_img)
        else:
            print("Warning: self.image is empty.")

    def prewitt(self):
        if self.image is not None:
            hx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            hy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

            #magnitude and direction
            magnitude = np.abs(hx) + np.abs(hy)
            direct = np.arctan(hy / hx)

            image1 = cv2.filter2D(self.image, -1, hx)
            image2 = cv2.filter2D(self.image, -1, hy)
            self.image = image1 + image2

            self.showImage(self.lbl_input_img, self.image)

            #self.createNamedWindow("prewitt_hx", image1)
            #self.createNamedWindow("prewitt_hy", image2)
            #self.createNamedWindow("prewitt", image3)

        else:
            print("Warning: self.image is empty.")


    def sobel(self):
        if self.image is not None:
            hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            hy = np.array([[1, 2, -1], [0, 0, 0], [-1, -2, -1]])

            magnitude = np.abs(hx) + np.abs(hy)
            direct = np.arctan(hy / hx)

            #image = cv2.Sobel(self.image, -1, hx, hy)

            image1 = cv2.filter2D(self.image, -1, hx)
            image2 = cv2.filter2D(self.image, -1, hy)
            self.image = image1 + image2

            self.showImage(self.lbl_input_img, self.image)
            #self.createNamedWindow("sobel_hx", image1)
            #self.createNamedWindow("sobel_hy", image2)
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
            plt.savefig('image/Threshold.png')
            plt.clf()
            Threshold_Image = cv2.imread('image/Threshold.png')
            self.showImage(self.lbl_input_img, Threshold_Image)
            # plt.show()
        else:
            print("Warning: self.image is empty.")

    def applyAdaptiveThreshold(self):
        # this function kill the program
        # do not run

        if self.image is not None:
            H = self.low_gauss(0.8)
            img = self.applyFilter(H)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)
            self.image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.showImage(self.lbl_input_img, self.image)
            #cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            #cv2.imshow('output', output)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
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
            # plt.imshow(img), plt.colorbar(), plt.show()

            plt.imshow(img), plt.colorbar()

            plt.savefig('image/Colorbar.png')
            plt.clf()
            Colorbar = cv2.imread('image/Colorbar.png')
            self.showImage(self.lbl_input_img, Colorbar)
        else:
            print("Warning: self.image is empty.")

    # def idontwanttosuffer4(self):
    def low_gauss(self, sigma = 1):
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

    def applyFilter(self, H):
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            G = np.fft.fftshift(np.fft.fft2(image))
            Ip = G * H
            im = np.abs(np.fft.ifft2(np.fft.fftshift(Ip)))
            return np.uint8(im)
        else:
            print("Warning: self.image is empty")

    def draw3D(self):
        hr = 4 / 2
        hc = 4 / 2

        x = np.arange(-hc, hc)
        y = np.arange(-hr, hr)

        [x, y] = np.meshgrid(x, y)
        mg = np.sqrt(x ** 2 + y**2)
        z = np.sin(mg)
        #print(mg)
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        # ax.plot_wireframe(x, y, z, color = 'black', linewidth = 0.8)
        # ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')
        surf = ax.plot_surface(x, y, z, cmap='coolwarm',
                        linewidth=0, antialiased=False)
        #ax = plt.axes(projection = '3d')(
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter(''))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('image/3D_diagram.png')
        plt.clf()
        Diagram_Image = cv2.imread('image/3D_diagram.png')
        self.showImage(self.lbl_input_img, Diagram_Image)
        # plt.show()

    def drawHistogram(self):
        img = self.image

        plt.hist(img.ravel(), 256, [0, 256])

        plt.savefig('image/Histogram.png')

        plt.clf()

        Histogram_Image = cv2.imread('image/Histogram.png')

        self.showImage(self.lbl_input_img, Histogram_Image)
    
    def denoise(self):
        # if self.image is not None:
        prob = 0.09
        self.output = np.zeros(self.image.shape, np.uint8)
        thres = 1 - prob
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    self.output[i][j] = 0
                elif rdn > thres:
                    self.output[i][j] = 255
                else:
                    self.output[i][j] = self.image[i][j]
        # return output
        # noise = np.zeros(self.image.shape, dtype=np.uint8)
        # cv2.randn(noise, 0, 255)
        # new_img = self.image + noise
        self.showImage(self.lbl_input_img, self.output)
        # print(self.output.shape)
        return self.output

    # def test_combobox(self):
    #     type_of_filer = self.cbox_blur.currentIndex()
    #     if type_of_filer == 0:
    #         self.Mean()


    def Mean(self):
        img = self.image.astype(np.float32) / 255
        blur = self.image.astype(np.float32) / 255
        # img = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg').astype(np.float32) / 255
        # blur = cv2.imread('C:/Users/Administrator/PycharmProjects/pythonProject/data/cats.jpg').astype(np.float32) / 255
        blur -= img.mean()
        blur /= img.std()
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Mean')
        plt.xticks([]), plt.yticks([])
        plt.savefig('image/Mean.png')
        plt.clf()
        Mean_Filter_Image = cv2.imread('image/Mean.png')
        self.showImage(self.lbl_input_img, Mean_Filter_Image)
        # plt.show()

    def Blur(self, kernel_x, kernel_y):
        img = self.image
        blur = cv2.blur(img, (kernel_x, kernel_y))
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.savefig('image/Blur.png')
        plt.clf()
        Blur_Filter_Image = cv2.imread('image/Blur.png')
        self.showImage(self.lbl_input_img, Blur_Filter_Image)

    def Median(self, kernel_z):
        img = self.denoise()
        blur = cv2.medianBlur(img, kernel_z)
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Median')
        plt.xticks([]), plt.yticks([])
        plt.savefig('image/Median.png')
        plt.clf()
        Median_Filter_Image = cv2.imread('image/Median.png')
        self.showImage(self.lbl_input_img, Median_Filter_Image)

    def Gauss(self, kernel_x, kernel_y):
        img = self.denoise()
        blur = cv2.GaussianBlur(img, (kernel_x, kernel_y), 0)
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Gauss')
        plt.xticks([]), plt.yticks([])
        plt.savefig('image/Gauss.png')
        plt.clf()
        Gauss_Filter_Image = cv2.imread('image/Gauss.png')
        self.showImage(self.lbl_input_img, Gauss_Filter_Image)

    def kmeans(self):
        img = self.image
        z = img.reshape((-1, 3))
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 7
        ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        self.showImage(self.lbl_input_img, res2)

if __name__ == "__main__":
    import os
    print(os.getenv("PYTHONPATH", "NONE"))
    app = QtWidgets.QApplication(sys.argv)
    tmp = Ui_MainWindow()
    app.exec_()
