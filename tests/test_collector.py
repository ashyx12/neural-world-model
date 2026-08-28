import numpy as np
from src.data.collector import collect_trajectory

def test_trajectory_structure():
    t = collect_trajectory("empty", seed=42, max_steps=10)
    assert 0 < len(t) <= 10
    x = t[0]
    assert set(x) == {"obs","action","next_obs","reward","terminated","truncated"}
    assert isinstance(x["action"], int)

def test_reproducible_collection():
    a = collect_trajectory("empty", seed=42, max_steps=20)
    b = collect_trajectory("empty", seed=42, max_steps=20)
    assert [x["action"] for x in a] == [x["action"] for x in b]
    for x, y in zip(a, b):
        assert np.array_equal(x["obs"]["image"], y["obs"]["image"])
        assert np.array_equal(x["next_obs"]["image"], y["next_obs"]["image"])
