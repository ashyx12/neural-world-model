import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.LazyLinear(latent_dim)
    def forward(self, x):
        return self.proj(self.net(x))

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3, image_size=7):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 2 * 2), nn.ReLU(),
            nn.Unflatten(1, (64, 2, 2)),
            nn.ConvTranspose2d(64, 32, 3, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(z)

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=64, action_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, z, action_onehot):
        return self.net(torch.cat([z, action_onehot], dim=-1))

class NeuralWorldModel(nn.Module):
    def __init__(self, action_dim=7, latent_dim=64):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim=latent_dim)
        self.decoder = CNNDecoder(latent_dim=latent_dim)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, action_dim=action_dim)
    def encode(self, obs):
        return self.encoder(obs)
    def decode(self, z):
        return self.decoder(z)
    def predict_latent(self, z, action_onehot):
        return self.dynamics(z, action_onehot)
