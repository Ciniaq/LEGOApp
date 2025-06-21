###########################################################################
#
# This script uses a YOLO model with SAHI slice-based predictions to detect
# objects in images from a specified directory. It handles large images by
# slicing them, then exports annotated results to an output folder.
# Used with break at the end to process only one image
#
###########################################################################

import os

import cv2
from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

model_path = "D:\Pycharm\\UnlitToBounds\\yolo_find_lego_model.pt"
model = YOLO(model_path, task="detect")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.8,
    device=0,  # or 'cuda:0'
)

source_directory_path = "D:\Pycharm\\UnlitToBounds\\real_images"
images = [f for f in os.listdir(source_directory_path) if f.endswith('.png') or f.endswith('.jpg')]
for file_name in images:
    image_path = os.path.join(source_directory_path, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    results.export_visuals(export_dir="demo_data/",
                           text_size=1,
                           rect_th=4,
                           hide_labels=True
                           )
    Image("demo_data/prediction_visual.png")
    break
