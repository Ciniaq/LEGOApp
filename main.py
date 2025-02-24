import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

csv_name = 'LegoParts.csv'
colors_dict = {}
actual_colors_dict = {}

images_path = r'D:\UELego\Lego\Saved\Screenshots\WindowsEditor'
images = {}

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
            images[name] = [file_name.split('_')[0] + '_original.jpg', file_name]

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
        image = Image.open(images_path+'\\'+masked)
        print(masked)
        width, height = image.size
        bool_array = np.zeros((height, width), dtype=bool)

        image_array = np.array(image)
        pixels = image_array.reshape(-1, image_array.shape[-1])
        unique_colors = np.unique(pixels, axis=0)

        for color in unique_colors:
            color_v = (int(color[0]), int(color[1]), int(color[2]), 255)
            actual_colors_dict[color_v] = colors_dict[find_nearest_color(color_v)[0]]


        for color in actual_colors_dict.keys():
            print(color, actual_colors_dict[color])
