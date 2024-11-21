import torch
import sys

# Importation conditionnelle pour éviter d'importer inutilement les bibliothèques
use_pygame = input("Voulez-vous utiliser Pygame (1) ou Matplotlib (2) ? [1/2]: ")

if use_pygame == "1":
    import pygame
elif use_pygame == "2":
    import matplotlib.pyplot as plt
else:
    print("Choix invalide. Veuillez relancer le script.")
    sys.exit(1)

# Importer le générateur depuis le fichier d'entraînement
from train import Generator  # Assurez-vous que ce chemin est correct

# Définir les dimensions
latent_dim = 100
image_dim = 28 * 28

# Charger le générateur sauvegardé
generator = Generator(latent_dim, image_dim)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

# Générer les images
z = torch.randn(16, latent_dim)  # Créer 16 vecteurs aléatoires
fake_images = generator(z).view(-1, 28, 28).detach()

if use_pygame == "1":
    # --- Mode Pygame ---
    pygame.init()
    screen_size = 400  # Taille de la fenêtre carrée
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Images Générées")

    # Transformer les images en format compatible avec Pygame
    images = [(img.numpy() * 255).astype('uint8') for img in fake_images]
    scaled_images = [pygame.transform.scale(pygame.surfarray.make_surface(img), (screen_size // 4, screen_size // 4)) for img in images]

    running = True
    while running:
        screen.fill((0, 0, 0))  # Fond noir

        # Afficher les images dans une grille 4x4
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                screen.blit(scaled_images[idx], (j * screen_size // 4, i * screen_size // 4))

        pygame.display.flip()  # Mettre à jour l'affichage

        # Gérer les événements pour fermer la fenêtre
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

elif use_pygame == "2":
    # --- Mode Matplotlib ---
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i].numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
