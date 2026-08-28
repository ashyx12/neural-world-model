# Neural World Model

An action-conditioned neural world model for visual MiniGrid environments.

## Project roadmap

1. Environment integration and reproducible trajectory collection
2. Baselines: persistence and state-space prediction
3. CNN encoder/decoder and latent representation
4. Action-conditioned latent dynamics
5. Multi-step imagination and evaluation
6. Data-efficiency, generalization, and environment-complexity experiments
7. Visualization and reproducible reporting
8. Optional extension: model-based planning / agentic control

The core project is completed before the planning/agentic extension is added.

## Environments

The initial implementation uses `MiniGrid-Empty-5x5-v0`. Dynamic Obstacles and DoorKey are registered in the environment factory when supported by the installed MiniGrid version.

## Structure

```
src/
  envs/          Environment factory
  data/          Trajectory collection and datasets
  models/        Encoder, decoder, latent dynamics, baselines
  training/      Training loops
  evaluation/    Metrics and experiment helpers
  planning/      Model-predictive control extension
  visualization/ Rollout and reconstruction plots
configs/         Experiment configuration
experiments/     Reproducible experiment entry points
tests/           Automated tests
demo/            Interactive/demo entry points
```

## Quick start

```bash
pip install -e .
pytest -q
```

Generate a small dataset:

```bash
python -m experiments.collect_data --env empty --seeds 100 --steps 100
```

Train the baseline world model:

```bash
python -m experiments.train --config configs/default.yaml
```

## Reproducibility

Datasets are generated from explicit environment seeds. Train/validation/test splits are seed-disjoint so evaluation measures generalization to unseen environment instances.

Generated datasets and experiment outputs are intentionally excluded from Git history.
