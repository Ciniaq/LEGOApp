import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained model
model_path = "D:\Pycharm\\UnlitToBounds\\runs2\\train\\exp\\weights\\last.pt"
model = YOLO(model_path, task="detect")

# Load the image
image_path = "D:\Pycharm\\UnlitToBounds\\real_images\\004.jpg"
# image_path = "D:\Pycharm\\UnlitToBounds\\dataset\\images\\val\\9-1503_original.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model.predict(image, conf=0.5)
# 416
annotated_image = results[0].plot(font_size=10, line_width=5)

plt.imshow(annotated_image)
plt.axis('off')
plt.show()
plt.imsave("output.png", annotated_image)
