import torch
import matplotlib.pyplot as plt
import os  # Pour vérifier si le fichier existe
from train import Generator  # Importation du générateur

print("Exécution de visualize_gan.py")  # Ceci doit s'afficher dans le terminal

# Définir les dimensions
latent_dim = 100  # Doit correspondre à la taille utilisée pendant l'entraînement
image_dim = 28 * 28

# Vérification si le fichier du générateur existe
if not os.path.exists("generator.pth"):
    print("Erreur: Le fichier 'generator.pth' n'existe pas. Aucune génération d'image possible.")
else:
    # Charger le générateur sauvegardé
    generator = Generator(latent_dim, image_dim)
    generator.load_state_dict(torch.load("generator.pth"))  # Charger les poids sauvegardés
    generator.eval()  # Mode évaluation (pas de mise à jour des gradients)

    # Générer des images aléatoires
    z = torch.randn(16, latent_dim)  # Créer 16 vecteurs aléatoires
    fake_images = generator(z).view(-1, 28, 28).detach()

    # Visualiser les images générées
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
