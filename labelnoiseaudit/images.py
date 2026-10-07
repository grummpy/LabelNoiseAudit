"""CPU image features: color histograms, a small thumbnail, and HOG.

No network and no pretrained weights. An optional embedding model lives in
``embeddings.py`` and stays off unless the caller opts in.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
CANVAS = 64
HIST_BINS = 8
THUMB = 16
ORIENTATIONS = 9
PIXELS_PER_CELL = 8


def feature_length() -> int:
    cells = CANVAS // PIXELS_PER_CELL
    blocks = max(cells - 1, 1)
    if cells >= 2:
        hog = blocks * blocks * (4 * ORIENTATIONS)
    else:
        hog = cells * cells * ORIENTATIONS
    return HIST_BINS * 3 + THUMB * THUMB + hog


def load_rgb(path: Path, size: int = CANVAS) -> Image.Image:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        rgb = transposed.convert("RGB")
        return rgb.resize((size, size), Image.Resampling.BILINEAR)


def color_histogram(rgb: Image.Image, bins: int = HIST_BINS) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.uint8)
    parts = []
    for channel in range(3):
        counts, _edges = np.histogram(array[:, :, channel], bins=bins, range=(0, 256))
        parts.append(counts.astype(np.float64))
    vector = np.concatenate(parts)
    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector


def thumbnail_features(rgb: Image.Image, size: int = THUMB) -> np.ndarray:
    gray = np.asarray(rgb.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64)
    return gray.ravel() / 255.0


def hog_features(
    rgb: Image.Image,
    orientations: int = ORIENTATIONS,
    pixels_per_cell: int = PIXELS_PER_CELL,
) -> np.ndarray:
    """Block-normalized histogram of oriented gradients on a grayscale canvas."""

    gray = np.asarray(rgb.convert("L"), dtype=np.float64) / 255.0
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    height, width = gray.shape
    cells_y = max(height // pixels_per_cell, 1)
    cells_x = max(width // pixels_per_cell, 1)
    histograms = np.zeros((cells_y, cells_x, orientations), dtype=np.float64)
    bin_width = 180.0 / orientations
    for cell_y in range(cells_y):
        for cell_x in range(cells_x):
            y0 = cell_y * pixels_per_cell
            x0 = cell_x * pixels_per_cell
            mag = magnitude[y0 : y0 + pixels_per_cell, x0 : x0 + pixels_per_cell]
            ang = orientation[y0 : y0 + pixels_per_cell, x0 : x0 + pixels_per_cell]
            if mag.size == 0:
                continue
            position = ang / bin_width
            lower = np.floor(position).astype(int) % orientations
            upper = (lower + 1) % orientations
            upper_weight = position - np.floor(position)
            lower_weight = 1.0 - upper_weight
            flat_mag = mag.ravel()
            np.add.at(histograms[cell_y, cell_x], lower.ravel(), lower_weight.ravel() * flat_mag)
            np.add.at(histograms[cell_y, cell_x], upper.ravel(), upper_weight.ravel() * flat_mag)
    if cells_y < 2 or cells_x < 2:
        vector = histograms.ravel()
        return vector / (np.linalg.norm(vector) + 1e-6)
    blocks = []
    for cell_y in range(cells_y - 1):
        for cell_x in range(cells_x - 1):
            block = histograms[cell_y : cell_y + 2, cell_x : cell_x + 2].ravel()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks)


def classic_features(path: Path) -> np.ndarray:
    rgb = load_rgb(path, CANVAS)
    vector = np.concatenate(
        [color_histogram(rgb), thumbnail_features(rgb), hog_features(rgb)]
    )
    if vector.shape != (feature_length(),):
        raise RuntimeError(f"Unexpected feature length {vector.shape[0]} for {path}")
    return vector


def thumbnail_png(path: Path, size: int = 96) -> bytes:
    rgb = load_rgb(path, size)
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return buffer.getvalue()


def glyph_png(pixels, size: int = 96) -> bytes:
    """Render one sklearn-digits row (0–16 ink) as a dark thumbnail."""

    array = np.asarray(pixels, dtype=float).reshape(8, 8)
    peak = float(array.max()) if array.size else 1.0
    scaled = np.clip(array / max(peak, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L").resize((size, size), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def iter_class_folder(root: Path) -> list[tuple[str, str]]:
    """Return ``(path, label)`` pairs from ``root/<class>/*``."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    records: list[tuple[str, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.name.startswith("."):
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            records.append((str(image_path.resolve()), class_dir.name))
    if not records:
        raise ValueError(
            f"No class subfolders with images under {root}. "
            "Use one folder per class, or a CSV with path and label columns."
        )
    return records


def read_image_csv(csv_path: Path) -> list[tuple[str, str]]:
    """Read a CSV with a path column and a label column."""

    import pandas as pd

    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    lookup = {str(column).lower(): column for column in frame.columns}
    path_name = next((lookup[key] for key in ("path", "filepath", "file", "filename", "image") if key in lookup), None)
    label_name = next((lookup[key] for key in ("label", "class", "target", "y") if key in lookup), None)
    if path_name is None or label_name is None:
        raise ValueError("Image CSV needs a path column and a label column")
    base = csv_path.parent
    records: list[tuple[str, str]] = []
    for raw_path, raw_label in zip(frame[path_name].astype(str), frame[label_name].astype(str), strict=True):
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Image listed in {csv_path.name} was not found: {candidate}")
        records.append((str(candidate), raw_label))
    if not records:
        raise ValueError(f"No rows in {csv_path}")
    return records


def safe_extract_zip(zip_path: Path, dest: Path, *, max_files: int = 20000, max_bytes: int = 200_000_000) -> Path:
    """Extract a zip and reject absolute paths and ``..`` members."""

    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    written = 0
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > max_files:
            raise ValueError(f"Zip has {len(members)} files; the limit is {max_files}")
        for info in members:
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Refusing to extract unsafe zip member {name!r}")
            target = (dest / name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise ValueError(f"Refusing to extract outside the destination: {name}")
            total += info.file_size
            if total > max_bytes:
                raise ValueError("Zip expands past the 200 MB limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                sink.write(source.read())
            written += 1
    if written == 0:
        raise ValueError("Zip did not contain any files")
    children = [path for path in dest.iterdir() if not path.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest
