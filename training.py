from ultralytics import YOLO

yaml_path = "D:\Pycharm\\UnlitToBounds\dataset\yolo11.yaml"
model_path = "D:\Pycharm\\UnlitToBounds\\trained_model1.pt"

model = YOLO(model_path, task="detect")

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
