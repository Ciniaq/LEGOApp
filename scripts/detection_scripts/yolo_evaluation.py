###########################################################################
#
# This script evaluates a YOLO model on a validation dataset using the SAHI library.
#
###########################################################################


import glob
import json
import os

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, CocoPrediction

# model_path = r"D:\Pycharm\UnlitToBounds\yolo_find_lego_model.pt"
model_path = r"D:\Pycharm\UnlitToBounds\epoch200.pt"  # 06092/200
val_folder = r"D:\Pycharm\UnlitToBounds\yolo_evaluation_data"

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.8,
    device="cuda:0",
)
image_paths = glob.glob(os.path.join(val_folder, "*.jpg"))

image_id = 0

coco = Coco()
coco.add_category(CocoCategory(id=0, name="lego"))
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    filename = os.path.basename(image_path)
    coco_image = CocoImage(file_name=filename, height=height, width=width)
    annotation_path = image_path.replace(".jpg", ".txt")
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            for line in f:
                cls, x_center, y_center, w, h = map(float, line.strip().split())
                x1 = (x_center - w / 2) * width
                y1 = (y_center - h / 2) * height
                x2 = (x_center + w / 2) * width
                y2 = (y_center + h / 2) * height
                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        category_id=0,
                        category_name='lego'
                    )
                )
    result = get_sliced_prediction(
        image,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    for object_prediction in result.object_prediction_list:
        coco_image.add_prediction(
            CocoPrediction(
                score=object_prediction.score,
                bbox=[
                    object_prediction.bbox.minx,
                    object_prediction.bbox.miny,
                    object_prediction.bbox.maxx - object_prediction.bbox.minx,
                    object_prediction.bbox.maxy - object_prediction.bbox.miny,
                ],
                category_id=0,
                category_name='lego'
            )
        )
    coco.add_image(coco_image)

for image in coco.images:
    for prediction in image.predictions:
        prediction.score = float(prediction.score.value)

with open("coco_gt.json", "w") as gt_file:
    json.dump(coco.json, gt_file)

with open("coco_pred.json", "w") as pred_file:
    json.dump(coco.prediction_array, pred_file)

coco_gt = COCO("coco_gt.json")
coco_pred = coco_gt.loadRes("coco_pred.json")

coco_evaluator = COCOeval(coco_gt, coco_pred, "bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
