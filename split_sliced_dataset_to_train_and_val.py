###########################################################################
#
# This script removes images that lack corresponding label files from the dataset,
# then randomly splits the remaining images (and their labels) into training and
# validation sets, moving the validation items to a designated folder
#
###########################################################################
import os
import random
import shutil

dataset_path = "D:\\Pycharm\\UnlitToBounds\\medium_dataset\\images\\train"
val_folder = "D:\\Pycharm\\UnlitToBounds\\medium_dataset\\images\\val"

# List all image files that do not have label files
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
               f.endswith('.png') and not os.path.exists(
                   os.path.join(dataset_path.replace('images', 'labels'), f.replace('.png', '.txt')))]

# all files that do not have labels - do not have legos on them - remove them
for file in image_files:
    if os.path.exists(file):
        os.remove(file)

image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
               f.endswith('.png')]

random.shuffle(image_files)
split_index = int(0.8 * len(image_files))  # 80% for training, 20% for validation
train_files = image_files[:split_index]
val_files = image_files[split_index:]

for file in val_files:
    if os.path.exists(file):
        shutil.move(file, val_folder)
        label_file = file.replace('images', 'labels').replace('.png', '.txt')
        if os.path.exists(label_file):
            shutil.move(label_file, val_folder.replace('images', 'labels'))
