import json

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QPushButton, QWidget, QGridLayout, QVBoxLayout, QLabel, QHBoxLayout

with open("../class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}


class ClassifyPopup(QWidget):
    def __init__(self, parent=None, croppedPixmap=None):
        super().__init__(parent)
        self.setWindowTitle("Classify lego")
        self.setWindowFlag(Qt.WindowType.Window)
        self.setFixedSize(750, 500)
        self.setStyleSheet("background-color: white;")

        self.main_layout = QVBoxLayout(self)
        self.grid_layout = QGridLayout()
        self.ok_layout = QHBoxLayout()

        icon_label = QLabel()
        if croppedPixmap:
            icon_label.setPixmap(croppedPixmap.scaledToHeight(100))
            icon_label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(icon_label)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.setFixedSize(100, 50)
        self.ok_btn.clicked.connect(self.ok_button_clicked)
        self.ok_layout.addWidget(self.ok_btn, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.ok_layout)

        buttons_per_row = 7
        for index, legoId in enumerate(class_to_idx.keys()):
            self.addLegoButton(legoId, index // buttons_per_row, index % buttons_per_row)

    def lego_button_clicked(self, name):
        print(f"{name} clicked")

    def ok_button_clicked(self):
        print(f"ok clicked")
        self.close()

    def addLegoButton(self, legoId, row, column):
        btn = QPushButton(legoId)
        btn.clicked.connect(lambda: self.lego_button_clicked(legoId))
        btn.setFixedSize(100, 50)

        pixmap = None
        try:
            pixmap = QPixmap(f'lego_icons/{legoId}.jpg')
        except Exception:
            pixmap = QPixmap(f'lego_icons/no_icon.jpg')
        btn.setIcon(QIcon(pixmap))
        btn.setIconSize(QSize(40, 40))

        self.grid_layout.addWidget(btn, row, column)
