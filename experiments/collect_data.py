import argparse
from pathlib import Path
from src.data.collector import collect_trajectories
from src.data.dataset import save_dataset, split_seeds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="empty")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="data/train.pkl")
    args = p.parse_args()
    train, _, _ = split_seeds(range(1, args.seeds + 1))
    save_dataset(collect_trajectories(args.env, train, args.steps), Path(args.out))

if __name__ == "__main__":
    main()
