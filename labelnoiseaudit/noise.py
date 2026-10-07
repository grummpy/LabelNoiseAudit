"""Inject synthetic label noise for the benchmark."""

from __future__ import annotations

import numpy as np


def inject_noise(
    labels,
    *,
    rate: float,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip labels and return ``(noisy_labels, was_flipped)``.

    ``uniform`` sends a flipped row to any other class.
    ``class_conditional`` sends every flipped row to the next class in
    sorted order, which is the same mistake for the whole class.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("noise rate must be between 0 and 1")
    if kind not in {"uniform", "class_conditional"}:
        raise ValueError("noise kind must be 'uniform' or 'class_conditional'")
    original = np.asarray(labels)
    classes = np.array(sorted(set(original.tolist()), key=str))
    if len(classes) < 2:
        raise ValueError("Noise injection needs at least two classes")
    position = {value: index for index, value in enumerate(classes.tolist())}
    rng = np.random.default_rng(seed)
    flip = rng.random(len(original)) < rate
    noisy = original.copy()
    mask = np.zeros(len(original), dtype=bool)
    for index in np.flatnonzero(flip):
        current = original[index]
        current_key = current.item() if isinstance(current, np.generic) else current
        if kind == "uniform":
            choices = [value for value in classes.tolist() if value != current_key]
            replacement = choices[int(rng.integers(0, len(choices)))]
        else:
            replacement = classes[(position[current_key] + 1) % len(classes)]
            replacement = replacement.item() if isinstance(replacement, np.generic) else replacement
        noisy[index] = replacement
        mask[index] = True
    return noisy, mask
