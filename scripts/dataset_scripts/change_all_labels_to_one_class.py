###########################################################################
#
# Script modifies the labels of the dataset to be one class (0)
# Needed for training a YOLO model for detection.
#
###########################################################################

import os

source_val_labels_path = "D:\Pycharm\\UnlitToBounds\\dataset\\labels\\val"
source_train_labels_path = "D:\Pycharm\\UnlitToBounds\\dataset\\labels\\train"
destination_val_labels_path = "D:\Pycharm\\UnlitToBounds\\dataset\\labelsDetect\\val"
destination_train_labels_path = "D:\Pycharm\\UnlitToBounds\\dataset\\labelsDetect\\train"


def modify_labels_to_be_one_class(source_directory_path, destination_directory_path):
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(source_directory_path) if f.endswith('.txt')]

    for file_name in txt_files:
        source_file_path = os.path.join(source_directory_path, file_name)
        destination_file_path = os.path.join(destination_directory_path, file_name)
        modified_lines = []

        # Read and modify each line
        with open(source_file_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                values[0] = "0"
                modified_lines.append(' '.join(values))

        # Write modified content back
        with open(destination_file_path, 'w') as file:
            file.write('\n'.join(modified_lines))


modify_labels_to_be_one_class(source_val_labels_path, destination_val_labels_path)
modify_labels_to_be_one_class(source_train_labels_path, destination_train_labels_path)
