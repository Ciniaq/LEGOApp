import json
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision import models
from ultralytics import YOLO

lock = threading.Lock()
frame = None
frame_gray = None
processed = True
overlay = np.zeros((1080, 1920), dtype=np.uint8)

yolo_model_path = "D:\Pycharm\\UnlitToBounds\\yolo_find_lego_model.pt"
yolo_model = YOLO(yolo_model_path, task="detect")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=yolo_model_path,
    confidence_threshold=0.8,
    device='cuda:0',
)

with open("../class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                        padding=model.conv1.padding, bias=False)
model.load_state_dict(torch.load("../resnet_grayscale_9.pth", map_location=device))
model = model.to(device)
model.eval()


def resize_and_pad_image(cropped_img, target_size=224):
    h, w, _ = cropped_img.shape

    if h > w:
        scale_factor = target_size / h
    else:
        scale_factor = target_size / w

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_img = cv2.resize(cropped_img, (new_w, new_h))

    top_padding = (target_size - new_h) // 2
    bottom_padding = target_size - new_h - top_padding
    left_padding = (target_size - new_w) // 2
    right_padding = target_size - new_w - left_padding

    padded_img = cv2.copyMakeBorder(resized_img, top_padding, bottom_padding, left_padding, right_padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img


def process_frame():
    global frame, overlay, processed
    while True:
        with lock:
            if frame is None:
                continue
            if processed:
                continue

            overlay.fill(0)
            results = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            print(len(results.object_prediction_list))
            for prediction in results.object_prediction_list:
                bbox = prediction.bbox
                cropped = frame[int(bbox.miny):int(bbox.maxy), int(bbox.minx):int(bbox.maxx)]
                ready_image = resize_and_pad_image(cropped)
                pil_image = transforms.ToPILImage()(ready_image)
                image_tensor = transform(pil_image).unsqueeze(0)
                input_tensor = image_tensor.to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence_predicted = confidence.item() * 100
                    predicted_class = idx_to_class[predicted.item()]
                    cv2.putText(overlay, f"{predicted_class} {confidence_predicted:.2f}%",
                                (int(bbox.minx), int(bbox.miny - 2),),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, 255, 2)
                    cv2.rectangle(overlay, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)),
                                  255,
                                  2)
            processed = True


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    processing_frame_thread = threading.Thread(target=process_frame)
    processing_frame_thread.daemon = True
    processing_frame_thread.start()

    print("Ready")
    while True:
        ret, new_frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        if lock.acquire(blocking=False):
            if frame_gray is None:
                frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                frame = new_frame.copy()
                processed = False
            else:
                gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(frame_gray, gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                diff = cv2.absdiff(frame_gray, gray)
                if cv2.countNonZero(thresh) > 20000:
                    print(cv2.countNonZero(thresh))
                    frame_gray = gray
                    frame = new_frame.copy()
                    print("Frame updated")
                    processed = False
            lock.release()

        blended = cv2.addWeighted(new_frame, 1, cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR), 1, 0)

        cv2.imshow("LegoDetection", blended)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
