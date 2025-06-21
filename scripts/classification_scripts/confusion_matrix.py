###########################################################################
#
# Creates confusion matrix for classification model,
# counts metrics such as precision, recall, F1-score
#
###########################################################################


import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.models import EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
with open("./class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# data_dir = "./dataset224"
data_dir = "./app/appDataset"
# data_dir = "D:\\Pycharm\\UnlitToBounds\\dataset_close\\classification"
# valset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
valset = datasets.ImageFolder(root=f"{data_dir}", transform=val_transform)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

# -----------------------------------
# model = models.resnet50(pretrained=False)
# model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
# model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
#                         padding=model.conv1.padding, bias=False)
# -----------------------------------

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels,
                                 kernel_size=model.features[0][0].kernel_size,
                                 stride=model.features[0][0].stride,
                                 padding=model.features[0][0].padding,
                                 bias=False)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 46)

# -----------------------------------
# model.load_state_dict(torch.load("./only_close_clear-77.pth", map_location=device)) # 51.37
# model.load_state_dict(torch.load("./resnet_grayscale_only_close_clear-35.pth", map_location=device))  # 52.61
model.load_state_dict(torch.load("./efficientnet_b0_epoch-17.pth", map_location=device))  # 55,65
# -----------------------------------

model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(valloader, start=1):
        if batch_idx % 10 == 0:
            print(f"Processing batch {int(batch_idx / 10)}/{int(len(valloader) / 10)}...")
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # -----------------------------------
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        # -----------------------------------
        # import torch.nn.functional as F
        #
        # probs = F.softmax(outputs, dim=1)
        # max_probs, preds = torch.max(probs, dim=1)
        # mask = max_probs > 0.6
        # filtered_preds = preds[mask]
        # filtered_labels = labels[mask]
        #
        # y_true.extend(filtered_labels.cpu().numpy())
        # y_pred.extend(filtered_preds.cpu().numpy())
        # -----------------------------------

cm = confusion_matrix(y_true, y_pred)
# np.fill_diagonal(cm, 0)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valset.classes)

fig, ax = plt.subplots(figsize=(18, 15))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.colorbar(im, ax=ax)

tick_marks = np.arange(len(valset.classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(valset.classes, rotation=90, ha='center', fontsize=12)
ax.set_yticklabels(valset.classes, rotation=0, fontsize=12)

for i in range(cm.shape[0]):
    row = ""
    for j in range(cm.shape[1]):
        row += f"{cm[i, j]}, "
        if cm[i, j] != 0:
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > 15 else "black", fontsize=8)
    print(row)

plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix_ffffff.png", dpi=300)
plt.close()

TP = []
FP = []
FN = []
TN = []

num_classes = len(valset.classes)

for i in range(num_classes):
    TP_i = cm[i][i]
    FP_i = cm[i].sum() - TP_i
    FN_i = cm[:, i].sum() - TP_i
    TN_i = cm.sum() - TP_i - FP_i - FN_i

    TP.append(TP_i)
    FP.append(FP_i)
    FN.append(FN_i)
    TN.append(TN_i)

    precision = TP_i / (TP_i + FP_i) if (TP_i + FP_i) > 0 else 0
    recall = TP_i / (TP_i + FN_i) if (TP_i + FN_i) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{valset.classes[i]}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

precision_per_class = []
recall_per_class = []
f1_per_class = []

for i in range(num_classes):
    TP = cm[i, i]
    FP = cm[i].sum() - TP
    FN = cm[:, i].sum() - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision_per_class.append(precision)
    recall_per_class.append(recall)
    f1_per_class.append(f1)

macro_precision = np.mean(precision_per_class)
macro_recall = np.mean(recall_per_class)
macro_f1 = np.mean(f1_per_class)

print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

total_correct = np.trace(cm)
total_samples = cm.sum()
total_accuracy = total_correct / total_samples

print(f"Total model accuracy: {total_accuracy:.4f}")
