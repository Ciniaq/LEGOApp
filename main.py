import math
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.measure import label, regionprops

csv_name = 'LegoParts.csv'
colors_dict = {}
actual_colors_dict = {}

images_path = r'D:\UELego\Lego\Saved\Screenshots\WindowsEditor'
images = {}

dataset_images_path = r'dataset/images'
dataset_labels_path = r'dataset/labels'


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

def fill_actual_colors_dict(image):
    actual_colors_dict.clear()
    image_array = np.array(image)
    pixels = image_array.reshape(-1, image_array.shape[-1])
    unique_colors = np.unique(pixels, axis=0)

    for color in unique_colors:
        color_v = (int(color[0]), int(color[1]), int(color[2]), 255)
        actual_colors_dict[color_v] = colors_dict[find_nearest_color(color_v)[0]]

def find_nearest_color(input_color):
    nearest_value = 1000
    nearest_color = None
    for color in colors_dict.keys():
        distance = np.linalg.norm(np.array(input_color) - np.array(color))
        if distance < nearest_value:
            nearest_value = distance
            nearest_color = color
    return nearest_color, float(nearest_value)

def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1.bbox
    x2_min, y2_min, x2_max, y2_max = box2.bbox

    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)
def distance_between_boxes(box1, box2, debug_output_array):
    x1_min, y1_min, x1_max, y1_max = box1.bbox
    x2_min, y2_min, x2_max, y2_max = box2.bbox

    dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
    dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))

    return math.sqrt(dx ** 2 + dy ** 2)

    # boxes overlap
    if left < 0 and right < 0 and top < 0 and bottom < 0:
        return 0

    return min(max(left, 0), max(right, 0), max(top, 0), max(bottom, 0))

if __name__ == '__main__':
    fill_colors_dict()
    fill_images_dict()

    for key, (original, masked) in images.items():
        image = Image.open(images_path + '\\' + masked)
        width, height = image.size

        debug_image = Image.open(images_path + '\\' + original)
        image_array = np.array(image)
        debug_output_array = np.array(debug_image)

        fill_actual_colors_dict(image)

        with open('dataset\\labels\\'+ original.split('.')[0] + '.txt', "w") as file:

            for color in actual_colors_dict.keys():
                if color == (0, 0, 0, 255):
                    continue
                mask = cv2.inRange(image_array, color, color)
                label_image = label(mask)
                regions = regionprops(label_image)
                filtered_regions = [region for region in regions if region.area >= 30]

                for region in filtered_regions:
                    minr, minc, maxr, maxc = region.bbox
                    if minc == 0:
                        continue
                    yolo_center_x = (minc + maxc) / 2 / width
                    yolo_center_y = (minr + maxr) / 2 / height
                    yolo_width = (maxc - minc) / width
                    yolo_height = (maxr - minr) / height
                    file.write(f"{actual_colors_dict[color]} {yolo_center_x:.6f} {yolo_center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")
                    cv2.rectangle(debug_output_array, (minc, minr), (maxc, maxr), color, 1)

                    print(f"{region.bbox} ({region.area})")

                for i in range(len(filtered_regions)):
                    for j in range(i + 1, len(filtered_regions)):
                        distance = distance_between_boxes(filtered_regions[i], filtered_regions[j],debug_output_array)
                        print(f"Boxes {i} and {j} are close: Distance={distance}")

        debug_image.save('dataset\\images\\'+original)
        image1 = Image.fromarray(debug_output_array)
        image1.show()
