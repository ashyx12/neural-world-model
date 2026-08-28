import torch

class WorldModelTrainer:
    def __init__(self, model, optimizer, recon_weight=1.0, dynamics_weight=1.0, reward_weight=0.1):
        self.model, self.optimizer = model, optimizer
        self.recon_weight, self.dynamics_weight, self.reward_weight = recon_weight, dynamics_weight, reward_weight

    def step(self, obs, action, next_obs, reward=None):
        self.model.train()
        out = self.model(obs, action, next_obs, reward)
        recon = torch.mean((out["next_pred"] - next_obs) ** 2)
        dyn = torch.mean((out["z_pred"] - out["z_target"].detach()) ** 2)
        reward_loss = torch.tensor(0.0, device=obs.device)
        if reward is not None:
            reward_loss = torch.mean((out["reward_pred"] - reward.float()) ** 2)
        loss = self.recon_weight * recon + self.dynamics_weight * dyn + self.reward_weight * reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "reconstruction": recon.item(), "dynamics": dyn.item(), "reward": reward_loss.item()}
