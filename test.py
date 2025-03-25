import os

import cv2
from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Load the trained model
# model_path = "D:\Pycharm\\UnlitToBounds\\runs\\train2\\exp\\weights\\last.pt"
# model_path = "D:\Pycharm\\UnlitToBounds\\saved_model05.pt"
model_path = "D:\Pycharm\\UnlitToBounds\\yolo_find_lego_model.pt"
model = YOLO(model_path, task="detect")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.8,
    device=0,  # or 'cuda:0'
)

# source_directory_path = "D:\Pycharm\\UnlitToBounds\\test_images"
source_directory_path = "D:\Pycharm\\UnlitToBounds\\real_images"
images = [f for f in os.listdir(source_directory_path) if f.endswith('.png') or f.endswith('.jpg')]
for file_name in images:
    # Load the image
    # image_path = "D:\Pycharm\\UnlitToBounds\\real_images\\004.jpg"
    # image_path = "D:\Pycharm\\UnlitToBounds\\dataset\\images\\val\\9-1503_original.png"
    image_path = os.path.join(source_directory_path, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = get_sliced_prediction(
        image_path,
        detection_model,
        # slice_height=416,
        # slice_width=416,
        # slice_height=816,
        # slice_width=816,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Perform object detection
    # results = model.predict(image, conf=0.3)
    # 416
    # annotated_image = results[0].plot(font_size=5, line_width=1)
    results.export_visuals(export_dir="demo_data/",
                           text_size=1,
                           rect_th=4,
                           hide_labels=True
                           )
    Image("demo_data/prediction_visual.png")
    # plt.imshow(annotated_image)
    # plt.axis('off')
    # plt.show()
    # plt.imsave("output.png", annotated_image)
    break
