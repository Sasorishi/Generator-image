# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import torch
# from models import Generator
# import base64
# from io import BytesIO
# from PIL import Image

# # Définir l'API
# app = FastAPI()

# try:
#     import torch
#     print("Torch loaded:", torch.__version__)
# except ImportError:
#     print("Torch not found. Please install it with 'pip install torch'.")
#     raise

# # Configurer les CORS pour autoriser les requêtes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Remplacez "*" par les URL spécifiques de votre front-end si nécessaire
#     allow_credentials=True,
#     allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
#     allow_headers=["*"],  # Autoriser tous les en-têtes
# )

# # Charger le modèle
# latent_dim = 100
# image_dim = 28 * 28  # Exemple pour MNIST
# generator = Generator(latent_dim, image_dim)
# generator.load_state_dict(torch.load("generator.pth"))
# generator.eval()

# # Requête pour générer des images
# class GenerateRequest(BaseModel):
#     num_images: int = 1  # Combien d'images à générer

# @app.post("/generate/")
# async def generate_images(request: GenerateRequest):
#     # Générer des images aléatoires
#     z = torch.randn(request.num_images, latent_dim)
#     # fake_images = generator(z).view(-1, 28, 28).detach()
#     fake_images = generator(z).view(-1, 28, 28).detach()

#     # Convertir les images en base64
#     base64_images = []
#     for img in fake_images:
#         pil_img = Image.fromarray(((img.numpy() + 1) * 127.5).astype("uint8"), mode="L")
#         buffered = BytesIO()
#         pil_img.save(buffered, format="PNG")
#         base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

#     return {"images": base64_images}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from models.gan import Generator
import base64
from io import BytesIO
from PIL import Image

# Définir l'API
app = FastAPI()

try:
    import torch
    print("Torch loaded:", torch.__version__)
except ImportError:
    print("Torch not found. Please install it with 'pip install torch'.")
    raise

# Configurer les CORS pour autoriser les requêtes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacez "*" par les URL spécifiques de votre front-end si nécessaire
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# Charger le modèle
latent_dim = 100
image_dim = 128 * 128 * 3  # Taille pour les images LFW (64x64 RGB)
generator = Generator(latent_dim, image_dim)
generator.load_state_dict(torch.load("generator_lfw.pth"))
generator.eval()

# Requête pour générer des images
class GenerateRequest(BaseModel):
    num_images: int = 1  # Combien d'images à générer

@app.post("/generate/")
async def generate_images(request: GenerateRequest):
    # Générer des images aléatoires
    z = torch.randn(request.num_images, latent_dim)
    fake_images = generator(z).view(-1, 128, 128, 3).detach()

    # Convertir les images en base64
    base64_images = []
    for img in fake_images:
        # Les valeurs générées par TANH sont entre -1 et 1, donc on les remet entre 0 et 255
        img = ((img.numpy() + 1) * 127.5).astype("uint8")

        # Créer une image PIL à partir des données RGB
        pil_img = Image.fromarray(img, mode="RGB")
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return {"images": base64_images}
