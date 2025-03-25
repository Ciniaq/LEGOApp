import json
import os
import random

import cv2
from PIL import Image

# Paths
coco_json_path = "mini_dataset/images/train/mini_coco_dataset.json_coco.json"
image_dir = "mini_dataset/images/train"

# Load COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Process each annotation
for img in coco_data["images"]:
    img_id = img["id"]
    img_width = img["width"]
    img_height = img["height"]
    img_filename = img["file_name"]

    image_path = os.path.join(image_dir, img_filename)
    img = cv2.imread(image_path)

    for ann in coco_data["annotations"]:
        if ann["image_id"] == img_id:
            x_min, y_min, bbox_width, bbox_height = map(int, ann['bbox'])

            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (x_min, y_min), (x_min + bbox_width, y_min + bbox_height), random_color, 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.show()
    break
