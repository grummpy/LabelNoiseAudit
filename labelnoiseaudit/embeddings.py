"""Optional pretrained image embeddings.

The default install does not include PyTorch. Nothing here runs, and no
weights are downloaded, unless the caller passes ``allow_download=True``
after installing ``requirements-embeddings.txt``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

Progress = Callable[[str], None]


def embeddings_status() -> dict[str, object]:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return {
            "installed": False,
            "detail": "torch and torchvision are not installed. The histogram and HOG features are the default.",
        }
    return {
        "installed": True,
        "detail": "Installed. Weights are downloaded only after you opt in.",
    }


def embed_paths(
    paths: Sequence[str],
    *,
    allow_download: bool,
    progress: Progress | None = None,
) -> np.ndarray:
    """ResNet-18 embeddings. Refuses to fetch weights unless opted in."""

    if not allow_download:
        raise RuntimeError(
            "Pretrained weights stay offline unless you opt in. "
            "Pass allow_download=True, or on the command line --allow-weight-download."
        )
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Embeddings need an extra install: pip install -r requirements-embeddings.txt"
        ) from exc

    if progress is not None:
        progress("Loading ResNet-18 weights (this can download them once)")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    vectors = []
    with torch.inference_mode():
        for index, raw_path in enumerate(paths, start=1):
            if progress is not None and (index == 1 or index % 25 == 0 or index == len(paths)):
                progress(f"Embedding image {index}/{len(paths)}")
            from PIL import Image

            with Image.open(Path(raw_path)) as image:
                rgb = image.convert("RGB")
                batch = transform(rgb).unsqueeze(0)
                vectors.append(model(batch).squeeze(0).numpy().astype(np.float64))
    if not vectors:
        raise ValueError("No images to embed")
    return np.vstack(vectors)
