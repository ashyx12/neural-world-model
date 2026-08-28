import numpy as np
import torch

def observation_tensor(obs, device="cpu"):
    image = obs["image"] if isinstance(obs, dict) else obs
    x = torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    return x / 255.0

def transition_tensors(transition, device="cpu"):
    return (
        observation_tensor(transition["obs"], device),
        torch.tensor([transition["action"]], dtype=torch.long, device=device),
        observation_tensor(transition["next_obs"], device),
    )
