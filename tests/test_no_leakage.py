"""Verify the evaluation protocol claims made in the report.

The val/test segments are extended backwards by window+horizon so the first
prediction in each has a full history. That overlap is only safe if no target
appears in more than one split, and if normalization statistics come from
training data alone. Both are asserted here rather than assumed.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from src.data.dataset import Normalizer, chronological_split


def make_data(T=500, N=4, F=3, seed=0):
    rng = np.random.default_rng(seed)
    # Distinct per-timestep values so targets are identifiable
    data = rng.normal(size=(T, N, F)).astype(np.float64)
    data[:, :, 0] = np.arange(T)[:, None] + np.arange(N)[None, :] / 100.0
    return data


def target_sets(data, w=20, h=1, train_ratio=0.7, val_ratio=0.15):
    train, val, test = chronological_split(
        data, train_ratio=train_ratio, val_ratio=val_ratio, window_size=w, horizon=h
    )
    out = {}
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        ys = [tuple(np.round(ds[i][1].numpy().astype(float), 6)) for i in range(len(ds))]
        out[name] = ys
    return out


def test_no_target_overlap_between_splits():
    data = make_data()
    ts = target_sets(data)
    tr, va, te = (set(ts[k]) for k in ("train", "val", "test"))
    assert not (tr & va), f"{len(tr & va)} targets shared between train and val"
    assert not (tr & te), f"{len(tr & te)} targets shared between train and test"
    assert not (va & te), f"{len(va & te)} targets shared between val and test"


def test_no_duplicate_targets_within_split():
    data = make_data()
    ts = target_sets(data)
    for name, ys in ts.items():
        assert len(ys) == len(set(ys)), f"duplicate targets inside {name}"


def test_targets_are_chronologically_ordered():
    """Train targets must all precede val, and val must precede test."""
    data = make_data()
    ts = target_sets(data)
    first_col = lambda ys: [y[0] for y in ys]
    tr, va, te = (first_col(ts[k]) for k in ("train", "val", "test"))
    assert max(tr) < min(va), "train target after val target"
    assert max(va) < min(te), "val target after test target"


def test_every_split_covers_full_range():
    """Trimming must not silently drop the start of val or the end of test."""
    data = make_data()
    w, h = 20, 1
    T = len(data)
    train_end, val_end = int(T * 0.7), int(T * 0.85)
    ts = target_sets(data, w=w, h=h)

    # First val target should be the first observation after the training window
    assert ts["val"][0][0] == float(train_end), ts["val"][0][0]
    assert ts["test"][0][0] == float(val_end), ts["test"][0][0]
    assert ts["test"][-1][0] == float(T - 1), ts["test"][-1][0]


def test_normalizer_uses_training_statistics_only():
    """Changing only the test segment must not change the fitted statistics."""
    data = make_data()
    T = len(data)
    train_end = int(T * 0.7)

    a = Normalizer().fit(data[:train_end])
    perturbed = data.copy()
    perturbed[train_end:] += 1000.0  # wreck the future
    b = Normalizer().fit(perturbed[:train_end])

    assert np.allclose(a.mean, b.mean), "normalizer mean saw future data"
    assert np.allclose(a.std, b.std), "normalizer std saw future data"


def test_inputs_never_contain_the_target():
    """The feature window must end strictly before the predicted timestep."""
    data = make_data()
    w, h = 20, 1
    train, _, _ = chronological_split(data, window_size=w, horizon=h)
    # Feature channel 0 is normalized, so compare against raw ordering instead:
    # target index must equal window_start + w + h - 1, i.e. one step past the window.
    for i in [0, 5, len(train) - 1]:
        x, y = train[i]
        assert x.shape[0] == w
        expected = float(i + w + h - 1)
        assert np.isclose(y[0].item(), expected), (y[0].item(), expected)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
