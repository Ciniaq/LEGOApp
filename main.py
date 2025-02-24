import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(5000)

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

def iterative_manage_pixel(start_x, start_y, color, image, bool_array, threshold=30):
    stack = [(start_x, start_y)]
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= image.size[0] or y >= image.size[1]:
            continue
        if bool_array[y][x]:
            continue
        pixel_color = image.getpixel((x, y))
        distance = np.linalg.norm(np.array(color) - np.array(pixel_color))
        if distance < threshold:
            image.putpixel((x, y), color)
            bool_array[y][x] = True
            stack.append((x - 1, y))  # Left
            stack.append((x + 1, y))  # Right
            stack.append((x, y - 1))  # Up
            stack.append((x, y + 1))  # Down

def rec_manage_pixel(x, y, color, image, bool_array, direction):
    if x < 0 or y < 0 or x >= image.size[0] or y >= image.size[1]:
        return
    if not bool_array[y][x]:
        pixel_color = image.getpixel((x, y))
        distance = np.linalg.norm(np.array(color) - np.array(pixel_color))
        if distance < 10:
            image.putpixel((x, y), color)
            bool_array[y][x] = True
            if direction != 'xup':
                rec_manage_pixel(x - 1, y, color, image, bool_array, 'xdown')
            if direction != 'xdown':
                rec_manage_pixel(x+1, y, color, image, bool_array, 'xup')
            if direction != 'yup':
                rec_manage_pixel(x, y - 1, color, image, bool_array, 'ydown')
            if direction != 'ydown':
                rec_manage_pixel(x, y+1, color, image, bool_array, 'yup')




if __name__ == '__main__':
    fill_colors_dict()
    fill_images_dict()

    # for item in colors_dict.items():
    #     print(item)
        # print(str(item[0][0])+","+str(item[0][1])+","+str(item[0][2]))

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

        #
        #
        # image.save(masked.split('.')[0] + '_modified2.bmp')
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        #
        # mono_image = Image.new('RGB', (width, height))
        # for y in range(height):
        #     for x in range(width):
        #         color = (255, 255, 255) if bool_array[y, x] else (0, 0, 0)
        #         mono_image.putpixel((x, y), color)
        # plt.imshow(mono_image)
        # plt.axis('off')
        # plt.show()
        # mono_image.save(masked.split('.')[0] + '_mono.bmp')
        # break
        #

