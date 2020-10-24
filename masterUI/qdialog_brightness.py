from PyQt5 import QtCore, QtGui, QtWidgets, uic

def foo():
    pass

class Ui_dialog_brightness(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('dialog_brightness.ui', self)
        self.show()
