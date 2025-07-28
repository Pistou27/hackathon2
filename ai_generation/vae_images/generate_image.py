import torch
from vae_model import VAE
import matplotlib.pyplot as plt

# Chargement
latent_dim = 20
model = VAE(latent_dim=latent_dim)
model.load_state_dict(torch.load("vae_mnist.pt", map_location=torch.device("cpu")))
model.eval()

# Génération
with torch.no_grad():
    z = torch.randn(1, latent_dim)
    sample = model.decode(z).view(28, 28)

plt.imshow(sample.numpy(), cmap="gray")
plt.title("Image générée par VAE")
plt.axis("off")
plt.show()