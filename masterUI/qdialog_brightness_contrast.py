from PyQt5 import QtCore, QtGui, QtWidgets, uic

class Ui_dialog_brightness_contrast(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        uic.loadUi('masterUI/ui/dialog_brightness_contrast.ui', self)
        '''preloaded'''

        '''find children'''
        self.lbl_brightness_value: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_brightness_value')
        self.lbl_contrast_value: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, 'lbl_contrast_value')
        self.hslider_brightness: QtWidgets.QSlider = self.findChild(QtWidgets.QSlider, 'hslider_brightness')
        self.hslider_contrast: QtWidgets.QSlider = self.findChild(QtWidgets.QSlider, 'hslider_contrast')
        self.btnbox: QtWidgets.QDialogButtonBox = self.findChild(QtWidgets.QDialogButtonBox, 'btnbox')

        self.hslider_contrast.valueChanged['int'].connect(lambda: self.lbl_contrast_value.setNum(self.hslider_contrast.value()/10))



        self.show()
