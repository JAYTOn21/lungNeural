import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PyQt6 import QtWidgets, QtGui
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWindow, QPixmap
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from networkx.convert_matrix import to_numpy_array

from about1 import aboutDialog
from main import Ui_MainWindow
from boot import predict

from torchvision import transforms
from PIL.ImageQt import ImageQt


def aboutfun():
    dialog = aboutDialog()
    dialog.exec()


def pil2pixmap(im):
    if im.mode == "RGB":
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
    elif im.mode == "RGBA":
        r, g, b, a = im.split()
        im = Image.merge("RGBA", (b, g, r, a))
    elif im.mode == "L":
        im = im.convert("RGBA")
    im2 = im.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)

    return pixmap


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.ui.pushButton.clicked.connect(self.open_dialog)
        self.ui.actionProgram.triggered.connect(aboutfun)

    def show(self):
        self.main_win.show()

    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(None, 'Open File', './', 'Images (*.png *.jpg *.jpeg)')
        if fname[0]:
            self.ui.lineEdit.setText(fname[0])
            self.ui.horizontalLayout_2.itemAt(self.ui.horizontalLayout_2.count() - 1).widget().deleteLater()
            pixmap = QPixmap(fname[0]).scaled(self.ui.labelIMG.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.ui.labelIMG.setPixmap(pixmap)
            img, res_data = predict(fname[0])
            pixmap2 = pil2pixmap(img).scaled(self.ui.labelIMG2.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.ui.labelIMG2.setPixmap(pixmap2)
            sc = Figure(facecolor='#1e1e1e')
            canvas = FigureCanvasQTAgg(sc)
            x = ['Пневмония', 'Здоровый', 'Туберкулез']
            ax = sc.figure.subplots()
            ax.set_facecolor('#202020')
            y = np.array(res_data)
            colors = ["#3873aa", "#2e992d", "#cd5555"]
            ax.bar(x, y, width=0.5, color=colors)
            ax.tick_params(colors='#AAAAAA')
            for spine in ax.spines.values():
                spine.set_edgecolor('#AAAAAA')
            self.ui.horizontalLayout_2.addWidget(canvas)
            canvas.draw()
            point = y.argmax()
            result = x[point]
            reses = ['Здоровый', 'Пневмония', 'Туберкулез', 'Подозрение на пневмонию', 'Подозрения на туберкулез']
            for i in range(len(y)):
                if i != point:
                    if abs(y[point] - y[i]) < 1:
                        if 'Пневмония' in [x[i], x[point]]:
                            result = reses[3]
                        elif 'Туберкулез' in [x[i], x[point]]:
                            result = reses[4]
            self.ui.lineEdit_2.setText(result)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())