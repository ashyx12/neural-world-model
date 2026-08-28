import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(latent_dim))

    def forward(self, x):
        return self.head(self.features(x))
