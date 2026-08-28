import itertools
import torch

@torch.no_grad()
def choose_action(model, observation, action_dim, horizon=3, device="cpu"):
    model.eval()
    z0 = model.encoder(observation.to(device))
    best_actions, best_return = None, float("-inf")
    for actions in itertools.product(range(action_dim), repeat=horizon):
        z = z0
        total = 0.0
        for a in actions:
            action = torch.tensor([a], device=device)
            z = model.dynamics(z, action)
            total += model.reward_head(z).mean().item()
        if total > best_return:
            best_return, best_actions = total, actions
    return best_actions[0]
