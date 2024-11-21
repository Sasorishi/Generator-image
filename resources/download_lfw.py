from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Préparation des transformations : convertit les images en tenseurs, les redimensionne et les normalise
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionner les images à 64x64 (ou autre taille souhaitée)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation pour les 3 canaux RGB
])

# Télécharger LFW (Labeled Faces in the Wild)
dataset = datasets.LFWPeople(
    root="./data",      # Chemin où le dataset sera stocké
    split="train",      # Utilise les données d'entraînement
    download=True,      # Télécharge le dataset si non présent
    transform=transform # Applique les transformations
)

# Charger les données dans un DataLoader pour les parcourir par batch
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Afficher un exemple pour vérifier
examples = iter(dataloader)
images, labels = next(examples)
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
