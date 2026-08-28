from src.data.collector import collect_trajectories

def collect_unseen_seed_set(env_name, start_seed, count, max_steps=100, observation="rgb"):
    return collect_trajectories(env_name, range(start_seed, start_seed + count), max_steps, observation)
