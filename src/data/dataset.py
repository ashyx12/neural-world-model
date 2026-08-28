from pathlib import Path
import pickle

def save_dataset(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dataset, f)

def load_dataset(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)

def split_seeds(seeds, train_ratio=0.8, val_ratio=0.1):
    seeds = sorted({int(s) for s in seeds})
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Invalid split ratios.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    n = len(seeds)
    a = int(n * train_ratio)
    b = a + int(n * val_ratio)
    return seeds[:a], seeds[a:b], seeds[b:]
