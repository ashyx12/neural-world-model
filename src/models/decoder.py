import torch.nn as nn

class CNNDecoder(nn.Module):
    """Decode a latent vector into a 40x40 RGB observation."""

    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * 5 * 5),
            nn.ReLU(),
            nn.Unflatten(1, (256, 5, 5)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)
