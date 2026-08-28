from src.envs.factory import make_env

def collect_trajectory(env_name="empty", seed=None, max_steps=100):
    env = make_env(env_name, seed=seed)
    if seed is not None:
        env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    trajectory = []
    for _ in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append({
            "obs": obs,
            "action": int(action),
            "next_obs": next_obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        obs = next_obs
        if terminated or truncated:
            break
    env.close()
    return trajectory

def collect_trajectories(env_name="empty", seeds=(), max_steps=100):
    return {int(s): collect_trajectory(env_name, int(s), max_steps) for s in seeds}
