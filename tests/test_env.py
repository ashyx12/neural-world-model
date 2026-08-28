import pytest
from src.envs.factory import make_env

def test_empty_environment():
    env = make_env("empty", seed=42)
    obs, _ = env.reset(seed=42)
    assert "image" in obs
    assert obs["image"].ndim == 3
    assert env.action_space.n > 0
    env.close()

def test_unknown_environment():
    with pytest.raises(ValueError):
        make_env("missing")
