import torch

@torch.no_grad()
def rollout(model, obs, actions):
    model.eval()
    z = model.encoder(obs)
    predictions = []
    for action in actions:
        z = model.dynamics(z, action)
        predictions.append(model.decoder(z))
    return predictions

@torch.no_grad()
def rollout_errors(model, obs, actions, targets):
    return [torch.mean((p - t) ** 2).item() for p, t in zip(rollout(model, obs, actions), targets)]
