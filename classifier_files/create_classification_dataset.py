###########################################################################
#
# This script creates classification dataset from YOLO annotations.
# It crops the images based on the YOLO annotations and resizes them to a fixed size.
#
###########################################################################

import os

import cv2

main_output_folder = "/home/macierz/s180439/close/dataset_close_class/train"
labels_path = "/home/macierz/s180439/close/dataset_close/labels/train"
images_path = "/home/macierz/s180439/close/dataset_close/images/train"
labels_file_path = "/home/macierz/s180439/close/dataset_close/labels.txt"
yoloID_2_legoID_dict = {}  # <yolo_id, lego_id>


def load_yolo_annotations(annotations_file, img_width, img_height):
    boxes = []
    with open(annotations_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes


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


def process_image(image_path, annotations_path, main_output_folder, file_name_with_png):
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    boxes = load_yolo_annotations(annotations_path, img_width, img_height)

    for idx, (class_id, x_min, y_min, x_max, y_max) in enumerate(boxes):
        current_sub_folder = f"{main_output_folder}/{str(yoloID_2_legoID_dict[class_id])}"
        os.makedirs(current_sub_folder, exist_ok=True)

        file_name = file_name_with_png.split(".")[0]

        cropped_img = img[y_min:y_max, x_min:x_max]
        processed_img = resize_and_pad_image(cropped_img)

        output_path = os.path.join(current_sub_folder, f"{file_name}_{idx + 1}.png")
        cv2.imwrite(output_path, processed_img)


def init_dict():
    with open(labels_file_path, "r") as labels_file:
        for line in labels_file:
            yolo_id, lego_id = line.strip().split(":")
            yoloID_2_legoID_dict[int(yolo_id.strip())] = int(float(lego_id.strip()))


if __name__ == '__main__':
    init_dict()
    counter = 0
    for file_name in os.listdir(images_path):
        image_path = os.path.join(images_path, file_name)
        annotations_path = os.path.join(labels_path, file_name.replace(".png", ".txt"))
        process_image(image_path, annotations_path, main_output_folder, file_name)
        print(f"{counter}/{len(os.listdir(images_path))}")
        counter += 1
        # break
