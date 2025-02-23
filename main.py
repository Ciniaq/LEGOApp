import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

csv_name = 'LegoParts.csv'
colors_dict = {}

images_path = r'D:\UELego\Lego\Saved\Screenshots\WindowsEditor'
images = {}

def fill_colors_dict():
    lego_parts_df = pd.read_csv(csv_name)
    legos = lego_parts_df[['---', 'UnlitColor']]
    colors_dict[(0, 0, 0)] = 0
    for index, row in legos.iterrows():
        color_components = row['UnlitColor'].strip("()").split(",")
        color_dict = {k: int(v) for k, v in (item.split("=") for item in color_components)}
        rgb_value = (color_dict['R'], color_dict['G'], color_dict['B'])
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
    return nearest_color

if __name__ == '__main__':
    fill_colors_dict()
    fill_images_dict()

    # for item in colors_dict.items():
    #     print(item)
        # print(str(item[0][0])+","+str(item[0][1])+","+str(item[0][2]))

    for key, (original, masked) in images.items():
        print(masked)
        image = Image.open(images_path+'\\'+masked)
        width, height = image.size
        for x in range(width):
            for y in range(height):
                color = image.getpixel((x, y))
                if(color[0] + color[1] + color[2] < 45):
                    image.putpixel((x, y), (0, 0, 0))
                else:
                    image.putpixel((x, y), find_nearest_color(color))
        image.save(masked.split('.')[0] + '_modified.jpg')
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        break


