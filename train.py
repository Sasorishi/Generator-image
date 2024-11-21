import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
import argparse

# 1. Hyperparamètres
latent_dim = 100
image_dim = 28 * 28  # Taille par défaut (pour MNIST, sera modifié dynamiquement)
batch_size = 64
lr = 0.0002
epochs = 50

# 2. Préparer le parser pour choisir le dataset
parser = argparse.ArgumentParser(description="Choisissez le dataset pour l'entraînement du GAN.")
parser.add_argument("--dataset", type=str, choices=["mnist", "celebA", "lfw"], default="mnist",
                    help="Dataset à utiliser: 'mnist', 'celebA' ou 'lfw'")
args = parser.parse_args()

# 3. Préparation des données
if args.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    image_dim = 28 * 28  # Taille des images MNIST aplaties
elif args.dataset == "celebA":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CelebA(root="./data", split="train", transform=transform, download=True)
    image_dim = 128 * 128 * 3  # Taille des images CelebA aplaties (64x64x3)
elif args.dataset == "lfw":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.LFWPeople(root="./data", split="train", transform=transform, download=True)
    image_dim = 128 * 128 * 3  # Taille des images LFW aplaties (64x64x3)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Initialisation des modèles et optimisateurs
generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)
optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_D = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# 5. Code d'entraînement (uniquement si ce fichier est exécuté directement)
if __name__ == "__main__":
    for epoch in range(epochs):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.view(-1, image_dim)
            batch_size = real_imgs.size(0)

            # Labels pour réel et faux
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Entraînement du Discriminateur
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            real_loss = loss_fn(discriminator(real_imgs), real_labels)
            fake_loss = loss_fn(discriminator(fake_imgs.detach()), fake_labels)
            loss_D = real_loss + fake_loss
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Entraînement du Générateur
            gen_loss = loss_fn(discriminator(fake_imgs), real_labels)
            optim_G.zero_grad()
            gen_loss.backward()
            optim_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {gen_loss.item():.4f}")

    # Sauvegarder les modèles entraînés
    torch.save(generator.state_dict(), f"generator_{args.dataset}.pth")
    torch.save(discriminator.state_dict(), f"discriminator_{args.dataset}.pth")
