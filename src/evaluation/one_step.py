import torch

@torch.no_grad()
def one_step_mse(model, observations, actions, next_observations):
    model.eval()
    pred = model(observations, actions)["next_pred"]
    return torch.mean((pred - next_observations) ** 2).item()
