import argparse
import torch
from src.data.dataset import load_dataset
from src.models.world_model import NeuralWorldModel
from src.training.loops import train_on_trajectories

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.pkl")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", default="checkpoints/world_model.pt")
    args = p.parse_args()

    dataset = load_dataset(args.data)
    trajectories = list(dataset.values()) if isinstance(dataset, dict) else dataset
    model = NeuralWorldModel(action_dim=args.action_dim, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = train_on_trajectories(model, trajectories, optimizer, args.epochs)
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print({"epochs": args.epochs, "loss": history[-1], "checkpoint": args.checkpoint})

if __name__ == "__main__":
    main()
