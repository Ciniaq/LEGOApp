###########################################################################
#
# This script counts annotations in YOLO format .txt files.
# For statistical purposes.
#
###########################################################################


import os

labels_path = r'./dataset/full_labels/train'
total_lines = 0
for root, _, files in os.walk(labels_path):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                total_lines += len(lines) - 1
print(f'Total number of lines in .txt files: {total_lines}')
print(f'Average number of lines per file: {total_lines / len(os.listdir(labels_path)):.2f}')
