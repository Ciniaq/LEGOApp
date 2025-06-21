###########################################################################
#
# Popup window for classifying lego sets, displays cropped image and allows
# user to save cropped image to the dataset folder
#
###########################################################################


import json
import os
from datetime import datetime

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QPushButton, QWidget, QGridLayout, QVBoxLayout, QLabel, QHBoxLayout

with open("../class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}


class ClassifyPopup(QWidget):
    def __init__(self, parent=None, croppedPixmap=None, predicted_name=None):
        super().__init__(parent)
        self.setWindowTitle("Classify lego")
        self.setWindowFlag(Qt.WindowType.Window)
        self.setFixedSize(750, 550)
        self.setStyleSheet("background-color: white;")
        self.clickedButtons = []
        self.buttonWidgets = []

        self.main_layout = QVBoxLayout(self)
        self.grid_layout = QGridLayout()
        self.ok_layout = QHBoxLayout()

        self.predicted_name = predicted_name
        self.textWidget = QLabel(f"Predicted lego is: {self.predicted_name}")
        self.main_layout.addWidget(self.textWidget, alignment=Qt.AlignCenter)

        self.icon_label = QLabel()
        if croppedPixmap:
            self.pixmap = croppedPixmap
            self.icon_label.setPixmap(self.pixmap.scaledToHeight(100))
            self.icon_label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.icon_label)
        else:
            self.pixmap = None
            self.icon_label.setPixmap(QPixmap("lego_icons/no_icon.jpg").scaledToHeight(100))
            self.icon_label.setAlignment(Qt.AlignCenter)
            self.icon_label.hide()
            self.main_layout.addWidget(self.icon_label)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.setFixedSize(100, 50)
        self.ok_btn.clicked.connect(self.ok_button_clicked)
        self.ok_layout.addWidget(self.ok_btn, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.ok_layout)

        buttons_per_row = 7
        for index, legoId in enumerate(class_to_idx.keys()):
            self.addLegoButton(legoId, index // buttons_per_row, index % buttons_per_row)

    def showPopup(self, croppedPixmap=None, predicted_name=None):
        self.clickedButtons.clear()
        self.predicted_name = predicted_name
        self.textWidget.setText(f"Predicted lego is: {self.predicted_name}")

        if croppedPixmap:
            self.pixmap = croppedPixmap
            self.icon_label.setPixmap(croppedPixmap.scaledToHeight(100))
            self.icon_label.show()
        else:
            self.pixmap = None
            self.icon_label.hide()

        for btn in self.buttonWidgets:
            btn.setStyleSheet("background-color: white;")
        self.show()

    def lego_button_clicked(self, name, button):
        print(f"{name} clicked")
        if name in self.clickedButtons:
            self.clickedButtons.remove(name)
            button.setStyleSheet("background-color: white;")
        else:
            self.clickedButtons.append(name)
            button.setStyleSheet("background-color: green;")

    def ok_button_clicked(self):
        if self.pixmap:
            mainDatasetFolder = "./appDataset"
            if not os.path.exists(mainDatasetFolder):
                os.makedirs(mainDatasetFolder)
            for legoType in self.clickedButtons:
                legoFolder = os.path.join(mainDatasetFolder, legoType)
                if not os.path.exists(legoFolder):
                    os.makedirs(legoFolder)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                self.pixmap.save(os.path.join(legoFolder, f"{timestamp}.png"))
            print(f"ok clicked, clicked buttons: {self.clickedButtons}")
        else:
            print("no pixmap, clicked buttons: ", self.clickedButtons)
        # self.close()
        self.hide()

    def addLegoButton(self, legoId, row, column):
        btn = QPushButton(legoId)
        btn.clicked.connect(lambda: self.lego_button_clicked(legoId, btn))
        btn.setFixedSize(100, 50)
        # if legoId == self.predicted_name:
        #     btn.setStyleSheet("color: red;")

        pixmap = None
        try:
            pixmap = QPixmap(f'lego_icons/{legoId}.jpg')
        except Exception:
            pixmap = QPixmap(f'lego_icons/no_icon.jpg')
        btn.setIcon(QIcon(pixmap))
        btn.setIconSize(QSize(40, 40))

        self.buttonWidgets.append(btn)
        self.grid_layout.addWidget(btn, row, column)
