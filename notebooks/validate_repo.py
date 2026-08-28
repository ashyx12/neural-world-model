# Kaggle validation for the current GitHub repository
# Run this notebook top-to-bottom. It validates the repository as-is; it does not train a large model.

!pip install -q -r /kaggle/working/neural-world-model/requirements.txt

import sys, os, math, pickle
sys.path.insert(0, "/kaggle/working/neural-world-model")
%cd /kaggle/working/neural-world-model

print("=== 1. Repository import check ===")
from src.data.collector import collect_trajectory, collect_trajectories
from src.data.dataset import save_dataset, load_dataset, split_seeds
from src.data.replay_buffer import ReplayBuffer
from src.data.tensors import observation_tensor, transition_tensors
from src.envs.factory import make_env, available_environments
from src.models.encoder import CNNEncoder
from src.models.decoder import CNNDecoder
from src.models.dynamics import ActionConditionedDynamics
from src.models.world_model import NeuralWorldModel
from src.models.baselines import PersistenceModel, PixelDynamicsModel
from src.training.trainer import WorldModelTrainer
from src.evaluation.one_step import one_step_mse
from src.evaluation.rollout import rollout, rollout_errors
from src.planning.mpc import choose_action
print("PASS: imports")

print("=== 2. Tests ===")
!pytest -q

print("=== 3. Environment validation ===")
for name in available_environments():
    env = make_env(name, seed=42)
    obs, _ = env.reset(seed=42)
    assert "image" in obs and obs["image"].ndim == 3
    assert env.action_space.n == 7
    env.close()
    print(f"PASS: {name}")

print("=== 4. RGB trajectory validation ===")
traj = collect_trajectory("empty", seed=42, max_steps=20, observation="rgb")
assert 0 < len(traj) <= 20
assert set(traj[0]) == {"obs","action","next_obs","reward","terminated","truncated"}
print("transitions:", len(traj))
print("image shape:", traj[0]["obs"]["image"].shape)

print("=== 5. Reproducibility ===")
a = collect_trajectory("empty", seed=42, max_steps=20, observation="rgb")
b = collect_trajectory("empty", seed=42, max_steps=20, observation="rgb")
assert [x["action"] for x in a] == [x["action"] for x in b]
print("PASS: deterministic seeded collection")

print("=== 6. Dataset round trip ===")
tmp = "/kaggle/working/validation.pkl"
save_dataset({42: traj}, tmp)
loaded = load_dataset(tmp)
assert len(loaded[42]) == len(traj)
print("PASS: save/load")

print("=== 7. Tensor conversion ===")
obs_t, action_t, next_t = transition_tensors(traj[0])
print(obs_t.shape, action_t.shape, next_t.shape)
assert obs_t.ndim == 4 and obs_t.shape[1] == 3
assert next_t.shape == obs_t.shape
print("PASS: tensors")

print("=== 8. World model forward pass ===")
import torch
model = NeuralWorldModel(action_dim=7, latent_dim=128)
obs = obs_t
action = action_t
next_obs = next_t
out = model(obs, action, next_obs, torch.tensor([traj[0]["reward"]], dtype=torch.float32))
for k, v in out.items():
    if torch.is_tensor(v): print(k, tuple(v.shape))
assert out["next_pred"].shape == next_obs.shape
assert out["z"].shape == out["z_pred"].shape == out["z_target"].shape
assert torch.isfinite(out["next_pred"]).all()
print("PASS: world model")

print("=== 9. One training step ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = WorldModelTrainer(model, optimizer)
metrics = trainer.step(obs, action, next_obs, torch.tensor([traj[0]["reward"]], dtype=torch.float32))
print(metrics)
assert all(math.isfinite(float(v)) for v in metrics.values())
print("PASS: optimizer/training step")

print("=== 10. Rollout ===")
actions = [int(x["action"]) for x in traj[:3]]
preds = rollout(model, obs, actions)
assert len(preds) == len(actions)
assert all(p.shape == next_obs.shape for p in preds)
print("PASS: rollout")

print("=== 11. MPC/planning ===")
chosen = choose_action(model, obs, action_dim=7, horizon=2)
assert 0 <= chosen < 7
print("chosen action:", chosen)
print("PASS: MPC")

print("\n=== VALIDATION COMPLETE ===")
