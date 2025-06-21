###########################################################################
#
# This script trains a ResNet classification model
#
###########################################################################

import json
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

data_dir = "./dataset224"
trainset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

valset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

valset2 = datasets.ImageFolder(root=f"./appDataset", transform=val_transform)
valloader2 = DataLoader(valset2, batch_size=32, shuffle=False)

class_to_idx = trainset.class_to_idx
with open("class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(trainset.classes))
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                        padding=model.conv1.padding, bias=False)

# model_to_load = "docs_test_2.pth"
# if os.path.exists(model_to_load):
#    model.load_state_dict(torch.load(model_to_load))
#    print(f"Model {model_to_load} loaded successfully.")
model = model.to(device)

class_counts = Counter(trainset.targets)
total_samples = len(trainset)
num_classes = len(trainset.classes)
class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(trainloader)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress = (batch_idx + 1) / total_batches * 100
        sys.stdout.write(f"\rEpoch {epoch + 1}, Progress: {progress:.2f}%")
        sys.stdout.flush()
    scheduler.step()
    print(f"\nEpoch {epoch + 1}, Loss: {running_loss / total_batches:.4f}")
    # Zapis modelu po kazdej epoce jakby mial sie w trakcie wywalic
    torch.save(model.state_dict(), f"docs_test_scheduler_{epoch + 1}.pth")

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader2:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy app: {100 * correct / total:.2f}%')

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Accuracy: {100 * correct / total:.2f}%')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
