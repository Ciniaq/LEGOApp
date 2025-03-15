import os

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained model
# model_path = "D:\Pycharm\\UnlitToBounds\\runs2\\train\\exp3\\weights\\last.pt"
model_path = "D:\Pycharm\\UnlitToBounds\\trained_model6.pt"
model = YOLO(model_path, task="detect")

source_directory_path = "D:\Pycharm\\UnlitToBounds\\test_images"
images = [f for f in os.listdir(source_directory_path) if f.endswith('.png')]
for file_name in images:
    # Load the image
    # image_path = "D:\Pycharm\\UnlitToBounds\\real_images\\004.jpg"
    # image_path = "D:\Pycharm\\UnlitToBounds\\dataset\\images\\val\\9-1503_original.png"
    image_path = os.path.join(source_directory_path, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model.predict(image, conf=0.7)
    # 416
    annotated_image = results[0].plot(font_size=5, line_width=1)

    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()
    # plt.imsave("output.png", annotated_image)
