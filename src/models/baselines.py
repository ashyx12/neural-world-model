import torch
import torch.nn as nn

class PersistenceModel:
    """Predict the next observation as the current observation."""
    def predict(self, obs):
        return obs

class StateSpaceBaseline(nn.Module):
    """Small fully-connected baseline over flattened observations."""
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, state, action_onehot):
        return self.net(torch.cat([state, action_onehot], dim=-1))
