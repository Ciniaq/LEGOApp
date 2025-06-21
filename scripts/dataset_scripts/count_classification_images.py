###########################################################################
#
# This script counts the number of files in the train and validation directories
# and calculates the average number of files per folder.
#
###########################################################################


import os

main_directory_path = r'/home/macierz/s180439/classifier_files/dataset224'
val_path = os.path.join(main_directory_path, 'val')
train_path = os.path.join(main_directory_path, 'train')
total_files = 0
print(os.getcwd())
print(main_directory_path)

print("VAL:")
for folder in os.listdir(val_path):
    folder_path = os.path.join(val_path, folder)
    if os.path.isdir(folder_path):
        file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        total_files += file_count
        print(f'Folder: {folder}, File Count: {file_count}')

print("TRAIN:")
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    if os.path.isdir(folder_path):
        file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        total_files += file_count
        print(f'Folder: {folder}, File Count: {file_count}')

print(f'Total number of files: {total_files}')
print(f'Average number of files per folder: {total_files / len(os.listdir(main_directory_path)):.2f}')
