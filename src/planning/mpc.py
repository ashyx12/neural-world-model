import itertools
import torch

def plan(model, observation, action_dim, horizon=5, device="cpu"):
    """Simple random-shooting MPC using model-predicted latent reconstruction loss."""
    with torch.no_grad():
        z0 = model.encode(observation.to(device))
        best_actions, best_score = None, float("inf")
        for actions in itertools.product(range(action_dim), repeat=horizon):
            z = z0
            for a in actions:
                onehot = torch.zeros(z.shape[0], action_dim, device=device)
                onehot[:, a] = 1.0
                z = model.predict_latent(z, onehot)
            score = torch.mean(z.square()).item()
            if score < best_score:
                best_score, best_actions = score, actions
    return best_actions[0]
