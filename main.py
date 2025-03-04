import math
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

# path to the csv file exported from DataTable in Unreal Engine
csv_name = 'LegoParts.csv'
colors_dict = {}
actual_colors_dict = {}
lego_id_dict = {}

# path to the unreal engine Screenshots folder
images_path = r'D:\UELego\Lego\Saved\Screenshots\WindowsEditor'
images = {}

dataset_images_path = r'dataset/images'
dataset_labels_path = r'dataset/labels'


def create_labels_file():
    lego_parts_df = pd.read_csv(csv_name)
    legos = lego_parts_df[['---']]
    i = 0
    with open('dataset\\labels\\labels.txt', "w") as labels_file:
        for index, row in legos.iterrows():
            # print(f"{i}: {row['---']}")
            labels_file.write(f"{i}: {row['---']}\n")
            lego_id_dict[row['---']] = i
            i += 1

    # string_foo = ""
    # for item in lego_id_dict.keys():
    #     string_foo += f"\"{item}\", "
    # print(f"[{string_foo}]")


def fill_colors_dict():
    lego_parts_df = pd.read_csv(csv_name)
    legos = lego_parts_df[['---', 'UnlitColor']]
    colors_dict[(0, 0, 0, 255)] = 0
    for index, row in legos.iterrows():
        color_components = row['UnlitColor'].strip("()").split(",")
        color_dict = {k: int(v) for k, v in (item.split("=") for item in color_components)}
        rgb_value = (color_dict['R'], color_dict['G'], color_dict['B'], 255)
        colors_dict[rgb_value] = row['---']


def fill_images_dict():
    for file_name in os.listdir(images_path):
        name = file_name.split('_')[0]
        if '_masked' in file_name:
            images[name] = [file_name.split('_')[0] + '_original.png', file_name]


def fill_actual_colors_dict(input_image):
    actual_colors_dict.clear()
    image_array = np.array(input_image)
    pixels = image_array.reshape(-1, image_array.shape[-1])
    unique_colors = np.unique(pixels, axis=0)

    for image_color in unique_colors:
        color_v = (int(image_color[0]), int(image_color[1]), int(image_color[2]), 255)
        actual_colors_dict[color_v] = colors_dict[find_nearest_color(color_v)[0]]


def find_nearest_color(input_color):
    nearest_value = 1000
    nearest_color = None
    for original_color in colors_dict.keys():
        distance = np.linalg.norm(np.array(input_color) - np.array(original_color))
        if distance < nearest_value:
            nearest_value = distance
            nearest_color = original_color
    return nearest_color, float(nearest_value)


def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)


def distance_between_boxes(box1, box2):
    if boxes_overlap(box1, box2):
        return 0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
    dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))

    return math.sqrt(dx ** 2 + dy ** 2)


def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    new_box = [min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max)]
    return new_box


def merge_all_boxes_in_array(regions_array):
    start_over = True
    output_array = [item.bbox for item in regions_array]

    while start_over:
        start_over = False
        for i in range(len(output_array)):
            for j in range(i + 1, len(output_array)):
                distance = distance_between_boxes(output_array[i], output_array[j])
                if distance < 15:
                    box1 = output_array[i]
                    box2 = output_array.pop(j)
                    output_array[i] = merge_boxes(box1, box2)
                    start_over = True
                    break
            if start_over:
                break
    return output_array


def create_YOLO_string(region, region_color):
    x1_min, y1_min, x1_max, y1_max = region

    yolo_center_x = (y1_min + y1_max) / 2 / width
    yolo_center_y = (x1_min + x1_max) / 2 / height
    yolo_width = (y1_max - y1_min) / width
    yolo_height = (x1_max - x1_min) / height
    return f"{lego_id_dict[actual_colors_dict[region_color]]} {yolo_center_x:.6f} {yolo_center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n"


if __name__ == '__main__':
    fill_colors_dict()
    fill_images_dict()
    create_labels_file()

    # for all images in the screenshot folder
    for key, (original, masked) in images.items():
        image = Image.open(images_path + '\\' + masked)
        print(masked)
        image_array = np.array(image)
        width, height = image.size

        # debug variables
        debug_image = Image.open(images_path + '\\' + original)
        debug_output_array = np.array(debug_image)

        fill_actual_colors_dict(image)

        # create file with labels
        with open('dataset\\labels\\' + original.split('.')[0] + '.txt', "w") as file:

            # for all colors in the image
            for color in actual_colors_dict.keys():
                if color == (0, 0, 0, 255):
                    continue

                # create mono mask for the color and find regions on it
                mask = cv2.inRange(image_array, color, color)
                label_image = label(mask)
                regions = regionprops(label_image)

                for region in regions:
                    print(region.area)

                avg_area = sum([region.area for region in regions]) / len(regions)
                print(f"{actual_colors_dict[color]}: {avg_area}")

                # filtered_regions = [region for region in regions if region.area >= 30]
                # merged_array = merge_all_boxes_in_array(filtered_regions)
                #
                # for region in merged_array:
                #     file.write(create_YOLO_string(region, color))
                #
                #     # debug image draw bounding boxes
                #     x1_min, y1_min, x1_max, y1_max = region
                #     cv2.rectangle(debug_output_array, (y1_min, x1_min), (y1_max, x1_max), color, 1)

        # debug_image.save('dataset\\images\\' + original)
        # image1 = Image.fromarray(debug_output_array)
        # image1.show()
        # image1.save("image_archive\\foo.png")
