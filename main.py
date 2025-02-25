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
                for region in regions:
                    minr, minc, maxr, maxc = region.bbox
                    if minc == 0:
                        continue
                    yolo_center_x = (minc + maxc) / 2 / width
                    yolo_center_y = (minr + maxr) / 2 / height
                    yolo_width = (maxc - minc) / width
                    yolo_height = (maxr - minr) / height
                    file.write(f"{actual_colors_dict[color]} {yolo_center_x:.6f} {yolo_center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")

                    cv2.rectangle(debug_output_array, (minc, minr), (maxc, maxr), color, 1)
                    # break
        debug_image.save('dataset\\images\\'+original)
        image1 = Image.fromarray(debug_output_array)
        image1.show()
        # image1.save(original.split('.')[0] + '_boxes.png')

            break
        break






