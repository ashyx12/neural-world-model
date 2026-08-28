import torch
from src.data.tensors import transition_tensors

def train_on_trajectories(model, trajectories, optimizer, epochs=1, device="cpu"):
    model.to(device)
    history = []
    for _ in range(epochs):
        losses = []
        for trajectory in trajectories:
            for transition in trajectory:
                obs, action, next_obs = transition_tensors(transition, device)
                reward = torch.tensor([transition["reward"]], dtype=torch.float32, device=device)
                out = model(obs, action, next_obs, reward)
                recon = torch.mean((out["next_pred"] - next_obs) ** 2)
                dyn = torch.mean((out["z_pred"] - out["z_target"].detach()) ** 2)
                reward_loss = torch.mean((out["reward_pred"] - reward) ** 2)
                loss = recon + dyn + 0.1 * reward_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        history.append(sum(losses) / max(1, len(losses)))
    return history
