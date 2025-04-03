import pickle
import random
import threading

import cv2
import numpy as np
import timm
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision import transforms
from ultralytics import YOLO

lock = threading.Lock()
frame = None
frame_gray = None
processed = True
overlay = np.zeros((1080, 1920, 3), dtype=np.uint8)

yolo_model_path = "D:\Pycharm\\UnlitToBounds\\yolo_find_lego_model.pt"
yolo_model = YOLO(yolo_model_path, task="detect")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=yolo_model_path,
    confidence_threshold=0.8,
    device='cuda:0',
)
model_path = "../vit_lego_2_1_1.pth"
model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=46)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
with open('../idx_to_class.pkl', 'rb') as f:
    idx_to_class = pickle.load(f)


def resize_and_pad_image(cropped_img, target_size=384):
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
                outputs = model(image_tensor)
                logits = outputs
                predicted_class = logits.argmax(-1).item()
                cv2.putText(overlay, str(idx_to_class[predicted_class]), (int(bbox.minx), int(bbox.miny),),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (random.randint(50, 300), random.randint(50, 300), random.randint(50, 300)), 3)
                cv2.rectangle(overlay, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (0, 255, 0),
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
                if cv2.countNonZero(thresh) > 8000:
                    frame_gray = gray
                    frame = new_frame.copy()
                    print("Frame updated")
                    processed = False
            lock.release()

        blended = cv2.addWeighted(new_frame, 1, overlay, 0.9, 0)

        cv2.imshow("LegoDetection", blended)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
