import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained model
model_path = "D:\Pycharm\\UnlitToBounds\\trained_model1.pt"
model = YOLO(model_path, task="detect")

# Load the image
image_path = "D:\Pycharm\\UnlitToBounds\\real_images\\004.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model.predict(image)
for r in results:
    print(r.boxes.data)

annotated_image = results[0].plot(font_size=10, line_width=5)
plt.imshow(annotated_image)
plt.axis('off')
plt.show()
