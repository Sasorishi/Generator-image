from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Préparation du dataset CelebA
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionner les images
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normaliser dans [-1, 1]
])

dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
