import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje danych
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Załadowanie własnego zbioru danych
data_dir = "./dataset224"
trainset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

valset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

# Zapisanie mapowania indeksów do nazw klas
class_to_idx = trainset.class_to_idx
with open("class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)

# Załadowanie modelu ResNet
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(trainset.classes))  # Dostosowanie do liczby klas
model = model.to(device)

# Definicja funkcji straty i optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
num_epochs = 5
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

    print(f"\nEpoch {epoch + 1}, Loss: {running_loss / total_batches:.4f}")
    torch.save(model.state_dict(), f"resnet_epoch_{epoch + 1}.pth")  # Zapis modelu po każdej epoce

# Testowanie modelu
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

print(f'Accuracy: {100 * correct / total:.2f}%')

# Zapis końcowego modelu
torch.save(model.state_dict(), "resnet_final.pth")
