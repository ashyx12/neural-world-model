import torch

def persistence_mse(observations, next_observations):
    return torch.mean((observations - next_observations) ** 2).item()

def rollout_horizon_errors(errors_by_horizon):
    return {int(k): float(v) for k, v in errors_by_horizon.items()}
