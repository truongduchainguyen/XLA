# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_kernel_size.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets, uic


class Ui_Dialog_kernel_size(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        uic.loadUi('ui/dialog_kernel_size.ui', self)
        '''preloaded'''

        '''find children'''
        self.spb_x: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, 'spb_x')
        self.spb_y: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, 'spb_y')
        self.spb_z: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, 'spb_z')
        self.cbox_blur: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, 'cbox_blur')

        self.show()

