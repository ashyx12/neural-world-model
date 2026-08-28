import torch.nn as nn

class PixelDynamicsModel(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.action = nn.Linear(action_dim, 3)

    def forward(self, image, action):
        import torch.nn.functional as F
        if action.ndim == 1:
            action = F.one_hot(action.long(), self.action_dim).float()
        bias = self.action(action).unsqueeze(-1).unsqueeze(-1)
        return self.net(image) + bias
