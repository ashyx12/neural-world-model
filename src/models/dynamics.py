import torch
import torch.nn as nn

class ActionConditionedDynamics(nn.Module):
    def __init__(self, latent_dim=128, action_dim=7, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        if action.ndim == 1:
            action = torch.nn.functional.one_hot(action.long(), self.action_dim).float()
        return self.net(torch.cat([z, action], dim=-1))
