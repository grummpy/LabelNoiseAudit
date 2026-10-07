import numpy as np

from labelnoiseaudit.noise import inject_noise


def test_uniform_and_class_conditional_rates():
    labels = np.array(["a", "b", "c"] * 400)
    noisy, mask = inject_noise(labels, rate=0.2, kind="uniform", seed=9)
    assert abs(mask.mean() - 0.2) < 0.03
    flipped = noisy[mask]
    original = labels[mask]
    assert np.all(flipped != original)

    conditional, conditional_mask = inject_noise(labels, rate=0.2, kind="class_conditional", seed=9)
    mapping = {"a": "b", "b": "c", "c": "a"}
    for before, after in zip(labels[conditional_mask], conditional[conditional_mask], strict=True):
        assert after == mapping[before]


def test_noise_is_repeatable():
    labels = np.array([0, 1, 2, 3] * 50)
    first, first_mask = inject_noise(labels, rate=0.1, kind="uniform", seed=3)
    second, second_mask = inject_noise(labels, rate=0.1, kind="uniform", seed=3)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
