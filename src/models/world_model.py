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

    def forward(self, obs, action, next_obs=None):
        z = self.encoder(obs)
        z_pred = self.dynamics(z, action)
        next_pred = self.decoder(z_pred)
        out = {"z": z, "z_pred": z_pred, "next_pred": next_pred}
        if next_obs is not None:
            out["z_target"] = self.encoder(next_obs)
        return out

    @torch.no_grad()
    def imagine(self, obs, actions):
        self.eval()
        z = self.encoder(obs)
        frames = []
        for action in actions:
            z = self.dynamics(z, action)
            frames.append(self.decoder(z))
        return frames
