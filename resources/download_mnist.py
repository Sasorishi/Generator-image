from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Préparation des transformations : convertit les images en tenseurs et les normalise
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisation pour centrer les données
])

# Télécharger MNIST
dataset = datasets.MNIST(
    root="./data",      # Chemin où le dataset sera stocké
    train=True,         # Utilise les données d'entraînement
    download=True,      # Télécharge le dataset si non présent
    transform=transform # Applique les transformations
)

# Charger les données dans un DataLoader pour les parcourir par batch
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Afficher un exemple pour vérifier
examples = iter(dataloader)
images, labels = next(examples)
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
