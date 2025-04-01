import timm
import torch
from PIL import Image
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = datasets.ImageFolder(root='./dataset/train/', transform=transform)
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load the saved model
model_path = "vit_lego_2.pth"
model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=46)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the image
image_path = "./dataset/val/3660/9-1524_original_89.png"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Perform the prediction
outputs = model(image_tensor)
logits = outputs
predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {idx_to_class[predicted_class]}")
