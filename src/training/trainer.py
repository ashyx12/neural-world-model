import torch

class WorldModelTrainer:
    def __init__(self, model, optimizer, recon_weight=1.0, dynamics_weight=1.0):
        self.model, self.optimizer = model, optimizer
        self.recon_weight, self.dynamics_weight = recon_weight, dynamics_weight

    def step(self, obs, action, next_obs):
        self.model.train()
        out = self.model(obs, action, next_obs)
        recon = torch.mean((out["next_pred"] - next_obs) ** 2)
        dyn = torch.mean((out["z_pred"] - out["z_target"].detach()) ** 2)
        loss = self.recon_weight * recon + self.dynamics_weight * dyn
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "reconstruction": recon.item(), "dynamics": dyn.item()}
