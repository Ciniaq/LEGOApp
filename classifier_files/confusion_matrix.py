import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

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

data_dir = "./dataset224"
valset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                        padding=model.conv1.padding, bias=False)
model.load_state_dict(torch.load("./resnet_grayscale_augm_16.pth", map_location=device))
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
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
np.fill_diagonal(cm, 0)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valset.classes)

cm = confusion_matrix(y_true, y_pred)
# np.fill_diagonal(cm, 0)

fig, ax = plt.subplots(figsize=(18, 15))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=1000)

plt.colorbar(im, ax=ax)

tick_marks = np.arange(len(valset.classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(valset.classes, rotation=90, ha='center', fontsize=12)
ax.set_yticklabels(valset.classes, rotation=0, fontsize=12)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j] != 0:
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > 550 else "black", fontsize=8)

plt.title("Confusion Matrix (Capped at 1000)", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix_capped.png", dpi=300)
plt.close()
