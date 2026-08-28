import torch
import torch.nn as nn
from .encoder import CNNEncoder
from .decoder import CNNDecoder
from .dynamics import ActionConditionedDynamics

class NeuralWorldModel(nn.Module):
    def __init__(self, action_dim=7, latent_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = CNNEncoder(latent_dim=latent_dim)
        self.dynamics = ActionConditionedDynamics(latent_dim, action_dim)
        self.decoder = CNNDecoder(latent_dim)
        self.reward_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, action, next_obs=None, reward=None):
        z = self.encoder(obs)
        z_pred = self.dynamics(z, action)
        next_pred = self.decoder(z_pred)
        out = {"z": z, "z_pred": z_pred, "next_pred": next_pred,
               "reward_pred": self.reward_head(z_pred).squeeze(-1)}
        if next_obs is not None:
            out["z_target"] = self.encoder(next_obs)
        if reward is not None:
            out["reward_target"] = reward
        return out

    @torch.no_grad()
    def imagine(self, obs, actions):
        self.eval()
        z = self.encoder(obs)
        frames, rewards = [], []
        for action in actions:
            z = self.dynamics(z, action)
            frames.append(self.decoder(z))
            rewards.append(self.reward_head(z).squeeze(-1))
        return frames, rewards
