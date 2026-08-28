import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64 * 2 * 2), nn.ReLU(),
            nn.Unflatten(1, (64, 2, 2)),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.up(self.head(z))
