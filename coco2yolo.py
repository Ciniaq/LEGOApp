import json
import os

# Paths
coco_json_path = "mini_dataset/images/train/mini_coco_dataset.json_coco.json"
output_yolo_dir = "mini_dataset/labels/train"

# Load COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Process each annotation
for img in coco_data["images"]:
    img_id = img["id"]
    img_width = img["width"]
    img_height = img["height"]
    img_filename = img["file_name"]

    yolo_txt_path = os.path.join(output_yolo_dir, img_filename.replace(".png", ".txt"))

    with open(yolo_txt_path, "w") as yolo_file:
        for ann in coco_data["annotations"]:
            if ann["image_id"] == img_id:
                class_id = ann["category_id"] - 1  # COCO class IDs start at 1, YOLO starts at 0
                x_min, y_min, bbox_width, bbox_height = ann["bbox"]

                # Convert to YOLO format (normalized)
                x_center = (x_min + (bbox_width / 2)) / img_width
                y_center = (y_min + (bbox_height / 2)) / img_height
                bbox_width = bbox_width / img_width
                bbox_height = bbox_height / img_height

                # Write to YOLO annotation file
                yolo_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
