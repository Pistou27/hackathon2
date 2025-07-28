import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae_model import VAE
import torch.nn.functional as F

# Hyperparamètres
batch_size = 128
epochs = 5
latent_dim = 20
device = torch.device("cpu")

# Dataset
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)

# Modèle
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Entraînement
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

torch.save(model.state_dict(), "vae_mnist.pt")