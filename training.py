from ultralytics import YOLO

yaml_path = "/home/macierz/s180439/mini_dataset/mini_dataset/yolo11.yaml"
model_path = "/home/macierz/s180439/saved_model.pt"

model = YOLO("yolov8m.pt", task="detect")

model.train(
    data=yaml_path,
    epochs=30,
    imgsz=(416, 416),
    batch=4,
    device=0,
    patience=10,
    augment=True,
    mosaic=0,
    save_period=1,
    project="runs/train01",
    name="exp",
    degrees=0,
    scale=0.5,
    cache=True,
)

model.save(model_path)
