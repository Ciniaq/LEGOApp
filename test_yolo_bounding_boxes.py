###########################################################################
#
# This script displays bounding boxes on an image using YOLO format labels.
# Used for manual validation of bounding boxes and in general YOLO format.
#
###########################################################################

import random

import cv2
from PIL import Image


def draw_yolo_bboxes(image_path, label_path, class_names):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, bbox_width, bbox_height = map(float, values[1:])

        # Convert YOLO format to pixel values
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)

        # Draw the bounding box
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), random_color, 2)
        cv2.putText(img, class_names[class_id], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, random_color, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.show()
    pil_img.save(f"image_archive/{file_name}_bb.png")


file_name = "9-1641_original"
image_path = f"dataset/images/val/{file_name}.png"
label_path = f"dataset/full_labels/train/{file_name}.txt"
class_names = ["0", "3001", "2431", "3003", "3004", "3002", "3005", "3009", "3010", "3020", "3021", "3022", "3023",
               "3031",
               "3034", "3037", "3039", "3040", "3045", "3062", "3068", "3069", "3460", "3622", "3659", "3660", "3665",
               "3666", "3676", "3710", "3795", "4070", "4287", "6141", "6143", "6215", "11212", "30136", "30414",
               "32807", "44237", "50950", "54200", "60481", "85080", "85984", "93273"]
draw_yolo_bboxes(image_path, label_path, class_names)
