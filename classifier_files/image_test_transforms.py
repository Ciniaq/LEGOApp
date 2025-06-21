import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

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

transform_basic = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def show_tensor_image(tensor_img, kolor=False):
    img = tensor_img.clone().detach()
    img = img * 0.5 + 0.5
    if kolor:
        img = img.permute(1, 2, 0).numpy()
    else:
        img = img.squeeze(0)
    plt.imshow(img, cmap='gray')
    plt.axis('off')


img_path = 'D:\\Pycharm\\UnlitToBounds\\classifier_files\\dataset224\\3001\\9-1557_original_118.png'
img = Image.open(img_path)

plt.figure(figsize=(12, 4))
for i in range(6):
    if i == 0:
        # Original image
        plt.subplot(2, 3, i + 1)
        basic = transform_basic(img)
        show_tensor_image(basic, True)
        plt.title('Original')
        continue
    augmented = transform(img)
    plt.subplot(2, 3, i + 1)
    show_tensor_image(augmented)
plt.tight_layout()
plt.show()
