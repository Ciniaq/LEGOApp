import datetime

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Load the saved model
model_path = "vit_lego_2.pth"
model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=46)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='./dataset/train/', transform=transform)
val_dataset = datasets.ImageFolder(root="./dataset/val/", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Continue training
num_epochs = 10
for epoch in range(num_epochs):
    print(epoch)
    model.train()
    total_loss, correct = 0, 0
    i = 0
    print(datetime.datetime.now())
    for images, labels in train_loader:
        print(f"Postep training: {i}/{len(train_loader)}", end='\r', flush=True)
        i = i + 1
        images, labels = images.to("cuda"), labels.to("cuda")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    print()
    with torch.no_grad():
        i = 0
        for images, labels in val_loader:
            print(f"Postep val: {i}/{len(val_loader)}", end='\r', flush=True)
            i = i + 1
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")
    torch.save(model.state_dict(), f"vit_lego_2_{epoch}.pth")
torch.save(model.state_dict(), "vit_lego.pth")
