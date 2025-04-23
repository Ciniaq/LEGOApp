###########################################################################
#
# This script splits a dataset into training and validation sets. (classification dataset)
#
###########################################################################

import os
import random
import shutil

dataset_path = "./dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

os.makedirs(val_path, exist_ok=True)

val_split = 0.2  # 20%

for class_name in os.listdir(train_path):
    class_train_path = os.path.join(train_path, class_name)
    print(class_train_path)
    class_val_path = os.path.join(val_path, class_name)

    if os.path.isdir(class_train_path):
        os.makedirs(class_val_path, exist_ok=True)

        images = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]

        num_val_images = int(len(images) * val_split)
        val_images = random.sample(images, num_val_images)

        for img in val_images:
            shutil.move(os.path.join(class_train_path, img), os.path.join(class_val_path, img))
