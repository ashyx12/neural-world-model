import gymnasium as gym

ENVIRONMENTS = {
    "empty": "MiniGrid-Empty-5x5-v0",
    "dynamic_obstacles": "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "doorkey": "MiniGrid-DoorKey-5x5-v0",
}

def available_environments():
    return tuple(ENVIRONMENTS)

def make_env(name="empty", seed=None, render_mode=None):
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {name!r}. Available: {list(ENVIRONMENTS)}")
    env = gym.make(ENVIRONMENTS[name], render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
