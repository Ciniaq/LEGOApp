from ultralytics import YOLO

yaml_path = "D:\Pycharm\\UnlitToBounds\dataset\yolo11.yaml"
model_path = "D:\Pycharm\\UnlitToBounds\\trained_model3.pt"

model = YOLO(model_path, task="detect")

model.train(
    data=yaml_path,
    epochs=60,
    imgsz=(480, 270),
    batch=1,
    device="cpu",
    patience=10,
    augment=True,
    mosaic=0,
    save_period=1,
    project="runs2/train",
    name="exp2",
)

model.save(model_path)
