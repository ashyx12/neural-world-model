import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, image_size=7):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * image_size * image_size), nn.ReLU(),
            nn.Unflatten(1, (64, image_size, image_size)),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(z)
