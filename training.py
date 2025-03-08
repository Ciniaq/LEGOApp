from ultralytics import YOLO

yaml_path = "D:\Pycharm\\UnlitToBounds\dataset\yolo11.yaml"
model_path = "trained_model1.pt"

model = YOLO("yolov8n.pt", task="detect")

model.train(
    data=yaml_path,
    epochs=20,
    imgsz=(480, 270),
    batch=4,
    device="cpu",
    patience=10,
    augment=True,
)

model.save(model_path)
