###########################################################################
#
# This script creates a GUI application that uses a YOLO model to detect LEGO objects,
# and ResNet for classification in a video stream from a camera.
#
###########################################################################

import json
import sys
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QMouseEvent
from PySide6.QtWidgets import *
from PySide6.QtWidgets import QSizePolicy
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision import models
from ultralytics import YOLO

from ClassifyPopup import ClassifyPopup

lock = threading.Lock()
frame = None
frame_gray = None
processed = True
overlay = np.zeros((1080, 1920), dtype=np.uint8)
freeze = False

yolo_model_path = "D:\Pycharm\\UnlitToBounds\\yolo_find_lego_model.pt"
yolo_model = YOLO(yolo_model_path, task="detect")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=yolo_model_path,
    confidence_threshold=0.8,
    device='cuda:0',
)

with open("../class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                        padding=model.conv1.padding, bias=False)
model.load_state_dict(torch.load("../resnet_grayscale_augm_16.pth", map_location=device))
model = model.to(device)
model.eval()


class Lego_object:
    def __init__(self, bbox, class_id, confidence):
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence

    def draw(self):
        global overlay
        cv2.putText(overlay, f"{self.confidence:.2f}%",
                    (int(self.bbox.minx), int(self.bbox.miny - 2),),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
        cv2.rectangle(overlay, (int(self.bbox.minx), int(self.bbox.miny)), (int(self.bbox.maxx), int(self.bbox.maxy)),
                      255,
                      3)
        print(
            f"Coordinates: {int(self.bbox.minx)}, {int(self.bbox.miny)}, {int(self.bbox.maxx)}, {int(self.bbox.maxy)}")


class Lego_manager:
    def __init__(self):
        self.__lego_objects = []
        self.__lego_dict = {}

    def add(self, lego_object):
        self.__lego_objects.append(lego_object)
        if lego_object.class_id not in self.__lego_dict:
            self.__lego_dict[lego_object.class_id] = []
        self.__lego_dict[lego_object.class_id].append(lego_object)

    def clear(self):
        self.__lego_objects.clear()
        self.__lego_dict.clear()

    def get_classes(self):
        return list(self.__lego_dict.keys())

    def draw_class(self, class_id):
        if class_id in self.__lego_dict:
            for lego_object in self.__lego_dict[class_id]:
                lego_object.draw()
        else:
            print(f"No objects found for class {class_id}")

    def getLegoObject(self, x, y):
        for lego_object in self.__lego_objects:
            if lego_object.bbox.minx < x < lego_object.bbox.maxx and lego_object.bbox.miny < y < lego_object.bbox.maxy:
                return lego_object
        return None


found_legos = Lego_manager()


def resize_and_pad_image(cropped_img, target_size=224):
    h, w, _ = cropped_img.shape

    if h > w:
        scale_factor = target_size / h
    else:
        scale_factor = target_size / w

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_img = cv2.resize(cropped_img, (new_w, new_h))

    top_padding = (target_size - new_h) // 2
    bottom_padding = target_size - new_h - top_padding
    left_padding = (target_size - new_w) // 2
    right_padding = target_size - new_w - left_padding

    padded_img = cv2.copyMakeBorder(resized_img, top_padding, bottom_padding, left_padding, right_padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img


def process_frame(stop_event):
    global frame, overlay, processed, found_legos
    while not stop_event.is_set():
        with lock:
            if frame is None:
                continue
            if processed:
                continue

            overlay.fill(0)
            results = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            print(len(results.object_prediction_list))
            found_legos.clear()
            for prediction in results.object_prediction_list:
                if processed:
                    break
                bbox = prediction.bbox
                cropped = frame[int(bbox.miny):int(bbox.maxy), int(bbox.minx):int(bbox.maxx)]
                ready_image = resize_and_pad_image(cropped)
                pil_image = transforms.ToPILImage()(ready_image)
                image_tensor = transform(pil_image).unsqueeze(0)
                input_tensor = image_tensor.to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence_predicted = confidence.item() * 100
                    predicted_class = idx_to_class[predicted.item()]
                    if confidence_predicted > 60:
                        found_legos.add(Lego_object(bbox, predicted_class, confidence_predicted))
            processed = True


class ClickOverlay(QWidget):
    def __init__(self, parent=None, size=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background-color: rgba(255, 0, 0, 1);")
        self.resize(size)
        self.content_size = size

    def mousePressEvent(self, event: QMouseEvent):
        x = event.position().x() - (self.size().width() - self.content_size.width()) / 2
        y = event.position().y() - (self.size().height() - self.content_size.height()) / 2
        image_size = (1080, 1920)
        x_scaled = int(x * image_size[1] / self.content_size.width())
        y_scaled = int(y * image_size[0] / self.content_size.height())
        clicked_lego = found_legos.getLegoObject(x_scaled, y_scaled)
        cropped_pixmap = None
        if clicked_lego:
            cropped_image = frame[int(clicked_lego.bbox.miny):int(clicked_lego.bbox.maxy),
                            int(clicked_lego.bbox.minx):int(clicked_lego.bbox.maxx)]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = np.ascontiguousarray(cropped_image)
            height, width, channels = cropped_image.shape

            cropped_pixmap = QPixmap.fromImage(
                QImage(cropped_image.data, width, height, channels * width, QImage.Format_RGB888))
        print(
            f"Caught click at {clicked_lego.class_id if clicked_lego else 'None'}")
        popup = ClassifyPopup(self, cropped_pixmap)
        popup.show()

    def setContentSize(self, content_size):
        self.content_size = content_size


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEGO APP")

        self.layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label, 1)

        self.click_overlay = ClickOverlay(self.label, self.label.size())
        self.label.resizeEvent = lambda event: self.adjust_overlay()

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setFixedWidth(200)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area, 0)

        self.buttons = []
        # self.button = QPushButton("CLEAR", self.scroll_content)
        # self.button.clicked.connect(self.clearOverlay)
        # self.scroll_layout.addWidget(self.button)

        self.freeze_button = QPushButton("Freeze", self.scroll_content)
        self.freeze_button.clicked.connect(lambda: self.on_freeze_button_click())
        self.scroll_layout.addWidget(self.freeze_button)

        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stop_event = threading.Event()
        self.processing_frame_thread = threading.Thread(target=process_frame, args=(self.stop_event,))
        self.processing_frame_thread.daemon = True
        self.processing_frame_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def add_button(self, class_id):
        button = QPushButton(class_id, self.scroll_content)
        button.setProperty("class_id", class_id)
        try:
            pixmap = QPixmap(f'lego_icons/{class_id}.jpg')
        except Exception:
            pixmap = QPixmap(f'lego_icons/no_icon.jpg')
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(QSize(50, 50))
        button.clicked.connect(lambda: self.on_button_click(class_id))
        self.scroll_layout.addWidget(button)
        self.buttons.append(button)

    def refresh_buttons(self):
        global found_legos
        classes = found_legos.get_classes()
        buttonsToDelete = []
        for button in self.buttons:
            if button.property("class_id") not in classes:
                self.scroll_layout.removeWidget(button)
                button.deleteLater()
                buttonsToDelete.append(button)
        for button in buttonsToDelete:
            self.buttons.remove(button)

        for class_id in classes:
            if not any(button.property("class_id") == class_id for button in self.buttons):
                self.add_button(class_id)

    def update_frame(self):
        global frame, overlay, processed, frame_gray, freeze
        ret, new_frame = self.cap.read()
        if freeze:
            new_frame = frame.copy()
        if ret:
            if lock.acquire(blocking=False):
                if frame_gray is None:
                    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                    frame = new_frame.copy()
                    processed = False
                else:
                    gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(frame_gray, gray)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    if cv2.countNonZero(thresh) > 20000:
                        print(cv2.countNonZero(thresh))
                        frame_gray = gray
                        frame = new_frame.copy()
                        print("Frame updated")
                        processed = False
                lock.release()
            elif not processed:
                gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(frame_gray, gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                if cv2.countNonZero(thresh) > 20000:
                    print("force break")
                    processed = True
            blended = cv2.addWeighted(new_frame, 1, cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR), 1, 0)

            rgb_frame = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            q_img = QImage(rgb_frame.data, width, height, channels * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            label_width = self.label.width()
            label_height = self.label.height()

            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.refresh_buttons()

            self.label.setPixmap(scaled_pixmap)
            self.click_overlay.setContentSize(scaled_pixmap.size())
            self.label.setScaledContents(False)

    def adjust_overlay(self):
        if self.click_overlay:
            self.click_overlay.resize(self.label.size())

    def on_button_click(self, class_id):
        self.clearOverlay()
        found_legos.draw_class(class_id)

    def on_freeze_button_click(self):
        global freeze
        freeze = not freeze

    def clearOverlay(self):
        global overlay
        overlay.fill(0)

    def closeEvent(self, event):
        self.stop_event.set()
        self.processing_frame_thread.join()
        self.cap.release()
        event.accept()


app = QApplication(sys.argv)
window = CameraApp()
window.showMaximized()
sys.exit(app.exec())
