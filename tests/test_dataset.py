from src.data.dataset import split_seeds, save_dataset, load_dataset

def test_seed_split():
    train, val, test = split_seeds(range(1, 101))
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert sorted(train + val + test) == list(range(1, 101))

def test_round_trip(tmp_path):
    p = tmp_path / "data.pkl"
    data = {1: [{"action": 2}]}
    save_dataset(data, p)
    assert load_dataset(p) == data
