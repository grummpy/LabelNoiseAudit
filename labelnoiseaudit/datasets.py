"""Bundled datasets: sklearn's public sets and a generated text collection.

Iris, wine, and digits ship inside scikit-learn. The field notes and the
shape images are generated from a seed. Nothing here is personal data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TOPICS: dict[str, list[str]] = {
    "orbit": [
        "satellite", "apogee", "perigee", "thruster", "payload", "inclination",
        "groundtrack", "eclipse", "telemetry", "nozzle", "delta", "station",
        "attitude", "gyro", "solar", "array", "uplink", "downlink", "burn", "window",
    ],
    "garden": [
        "soil", "sprout", "watering", "mulch", "seedling", "compost", "trellis",
        "blossom", "pruner", "raised", "bed", "tomato", "pollinator", "drip",
        "hose", "shade", "cloth", "harvest", "row", "cover", "slug",
    ],
    "kitchen": [
        "simmer", "knife", "broth", "skillet", "dough", "whisk", "reduce",
        "garnish", "roast", "pan", "ladle", "mince", "fold", "butter", "stock",
        "herb", "sear", "rest", "plate", "zest",
    ],
    "harbor": [
        "dock", "buoy", "tide", "cleat", "fender", "mooring", "wake", "channel",
        "pier", "barnacle", "winch", "line", "pilot", "berth", "gangway",
        "anchor", "foghorn", "chart", "slip", "current",
    ],
}

FILLER = ["the", "and", "with", "from", "after", "before", "into", "while", "under", "across"]


def synthetic_text_frame(*, n_per_class: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for topic, vocab in TOPICS.items():
        for index in range(n_per_class):
            length = int(rng.integers(10, 16))
            other = [word for name, words in TOPICS.items() if name != topic for word in words]
            words = []
            for _ in range(length):
                roll = float(rng.random())
                if roll < 0.22:
                    words.append(str(rng.choice(FILLER)))
                elif roll < 0.40:
                    words.append(str(rng.choice(other)))
                else:
                    words.append(str(rng.choice(vocab)))
            rows.append(
                {
                    "note_id": f"{topic}-{index:03d}",
                    "note": " ".join(words),
                    "topic": topic,
                }
            )
    return pd.DataFrame(rows)


def load_builtin_frame(name: str, *, seed: int = 42) -> tuple[pd.DataFrame, str, str]:
    """Return a frame, the label column, and a thumbnail mode.

    Thumbnail mode is ``none``, ``glyphs`` (digits), or ``images``.
    """

    key = name.strip().lower()
    if key == "iris":
        from sklearn.datasets import load_iris

        bunch = load_iris()
        frame = pd.DataFrame(bunch.data, columns=[str(column) for column in bunch.feature_names])
        frame["species"] = [str(bunch.target_names[int(index)]) for index in bunch.target]
        return frame, "species", "none"
    if key == "wine":
        from sklearn.datasets import load_wine

        bunch = load_wine()
        frame = pd.DataFrame(bunch.data, columns=[str(column) for column in bunch.feature_names])
        frame["cultivar"] = [f"class_{int(index)}" for index in bunch.target]
        return frame, "cultivar", "none"
    if key == "digits":
        from sklearn.datasets import load_digits

        bunch = load_digits()
        columns = [f"pixel_{index}" for index in range(bunch.data.shape[1])]
        frame = pd.DataFrame(bunch.data, columns=columns)
        frame["digit"] = [str(int(value)) for value in bunch.target]
        return frame, "digit", "glyphs"
    if key in {"field_notes", "text"}:
        return synthetic_text_frame(seed=seed), "topic", "none"
    raise ValueError(f"Unknown built-in dataset {name!r}. Choose iris, wine, digits, or field_notes.")


def materialize_shapes(dest: Path, *, n_per_class: int = 24, seed: int = 42) -> Path:
    """Draw two classes of synthetic images into ``dest/<class>/``."""

    dest = Path(dest)
    rng = np.random.default_rng(seed)
    specs = {
        "sun": _draw_sun,
        "tile": _draw_tile,
    }
    for label, painter in specs.items():
        folder = dest / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(n_per_class):
            image = painter(rng)
            image.save(folder / f"{label}-{index:03d}.png")
    return dest


def _draw_sun(rng: np.random.Generator) -> Image.Image:
    sky = (
        int(rng.integers(20, 70)),
        int(rng.integers(80, 150)),
        int(rng.integers(170, 230)),
    )
    image = Image.new("RGB", (96, 96), sky)
    draw = ImageDraw.Draw(image)
    center = (int(rng.integers(40, 56)), int(rng.integers(40, 56)))
    radius = int(rng.integers(16, 24))
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        fill=(int(rng.integers(220, 255)), int(rng.integers(180, 220)), 40),
    )
    return image


def _draw_tile(rng: np.random.Generator) -> Image.Image:
    ground = (
        int(rng.integers(40, 80)),
        int(rng.integers(40, 80)),
        int(rng.integers(40, 80)),
    )
    image = Image.new("RGB", (96, 96), ground)
    draw = ImageDraw.Draw(image)
    margin = int(rng.integers(18, 30))
    draw.rectangle((margin, margin, 96 - margin, 96 - margin), fill=(200, 50, 40))
    return image


def benchmark_tasks(seed: int = 42):
    """Yield ``(name, kind, payload, label_column)`` for the benchmark runner.

    ``kind`` is ``matrix`` or ``text``. Matrix payload is ``(X, y)``.
    Text payload is a frame whose label column is replaced after noise injection.
    """

    for name in ("iris", "wine", "digits"):
        frame, label, _mode = load_builtin_frame(name, seed=seed)
        features = [column for column in frame.columns if column != label]
        yield name, "matrix", (frame[features].to_numpy(dtype=float), frame[label].to_numpy()), label
    notes = synthetic_text_frame(n_per_class=40, seed=seed)
    yield "synthetic_text", "text", notes, "topic"
