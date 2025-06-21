import json
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
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
# data_dir = "/home/macierz/s180439/close/dataset_close_class"
trainset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

valset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

valset2 = datasets.ImageFolder(root="./appDataset", transform=val_transform)
valloader2 = DataLoader(valset2, batch_size=32, shuffle=False)

with open("class_to_idx.json", "w") as f:
    json.dump(trainset.class_to_idx, f)

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels,
                                 kernel_size=model.features[0][0].kernel_size,
                                 stride=model.features[0][0].stride,
                                 padding=model.features[0][0].padding,
                                 bias=False)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(trainset.classes))

# model_to_load = "efficientnet_b0_gray.pth"
# if os.path.exists(model_to_load):
#     model.load_state_dict(torch.load(model_to_load))
#     print(f"Model {model_to_load} loaded successfully.")

model = model.to(device)

class_counts = Counter(trainset.targets)
total_samples = len(trainset)
num_classes = len(trainset.classes)
class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

num_epochs = 188
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

    # current_lr = scheduler.get_last_lr()[0]
    print(f"\nEpoch {epoch + 1} completed. Loss: {running_loss / total_batches:.4f}")

    torch.save(model.state_dict(), f"efficientnet_b0_epoch-{epoch + 1}.pth")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valloader2:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'AppDataset Accuracy: {100 * correct / total:.2f}%')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in valloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')
