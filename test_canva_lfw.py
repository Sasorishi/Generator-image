import torch
import sys
import matplotlib.pyplot as plt

# Importer le générateur depuis le fichier d'entraînement
from train import Generator  # Assurez-vous que ce chemin est correct

# Définir les dimensions
latent_dim = 100
image_dim = 128 * 128 * 3

# Charger le générateur sauvegardé
generator = Generator(latent_dim, image_dim)
generator.load_state_dict(torch.load("generator_lfw.pth"))
generator.eval()

# Générer les images
z = torch.randn(16, latent_dim)  # Créer 16 vecteurs aléatoires
fake_images = generator(z).view(-1, 3, 128, 128).detach()  # Redimensionner pour correspondre à l'image (C, H, W)

# --- Mode Matplotlib ---
fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # Grille 4x4 pour afficher les images
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[i].permute(1, 2, 0).numpy())  # Permuter pour passer de (C, H, W) à (H, W, C)
    ax.axis("off")
plt.tight_layout()
plt.show()
