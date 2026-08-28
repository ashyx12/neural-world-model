import argparse
from src.envs.factory import make_env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="empty")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    env = make_env(args.env, seed=args.seed, render_mode="human")
    obs, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
    env.close()

if __name__ == "__main__":
    main()
