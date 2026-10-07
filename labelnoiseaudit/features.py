"""Column kinds and a ColumnTransformer the out-of-fold loop fits per fold."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_LIKE = {
    "id",
    "row_id",
    "note_id",
    "sample_id",
    "path",
    "filepath",
    "file",
    "filename",
    "image",
    "img",
}


def infer_kind(series: pd.Series) -> str:
    """Classify one column as numeric, categorical, or text."""

    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    text = series.dropna().astype(str)
    if text.empty:
        return "categorical"
    n_rows = max(len(series), 1)
    nunique = int(text.nunique())
    avg_len = float(text.str.len().mean())
    space_frac = float(text.str.contains(r"\s", regex=True).mean())
    if avg_len >= 25 or (space_frac >= 0.6 and avg_len >= 12):
        return "text"
    if nunique > max(40, int(0.5 * n_rows)) and avg_len >= 8:
        return "text"
    return "categorical"


def infer_column_kinds(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {"numeric": [], "categorical": [], "text": []}
    for name in columns:
        kinds[infer_kind(frame[name])].append(name)
    return kinds


def suggest_label_column(frame: pd.DataFrame) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in ("label", "class", "target", "y", "topic", "species", "digit"):
        if candidate in lookup:
            return lookup[candidate]
    return str(frame.columns[-1])


def default_feature_columns(frame: pd.DataFrame, label_column: str) -> list[str]:
    chosen = []
    for column in frame.columns:
        name = str(column)
        if name == label_column or name.startswith("__lna_"):
            continue
        if name.lower() in ID_LIKE:
            continue
        chosen.append(name)
    return chosen


def normalize_kinds(kinds: dict[str, list[str]], feature_columns: list[str]) -> dict[str, list[str]]:
    """Keep only the selected feature columns, and reject overlaps."""

    allowed = set(feature_columns)
    cleaned = {"numeric": [], "categorical": [], "text": []}
    seen: set[str] = set()
    for kind in ("numeric", "categorical", "text"):
        for name in kinds.get(kind, []):
            if name not in allowed:
                continue
            if name in seen:
                raise ValueError(f"Column {name!r} is listed under more than one kind")
            seen.add(name)
            cleaned[kind].append(name)
    missing = [name for name in feature_columns if name not in seen]
    if missing:
        raise ValueError(f"No column kind for: {', '.join(missing)}")
    if not any(cleaned.values()):
        raise ValueError("Select at least one feature column")
    return cleaned


def prepare_frame(frame: pd.DataFrame, kinds: dict[str, list[str]]) -> pd.DataFrame:
    """Copy the feature columns into dtypes the transformers accept."""

    columns = kinds["numeric"] + kinds["categorical"] + kinds["text"]
    prepared = pd.DataFrame(index=frame.index)
    for name in kinds["numeric"]:
        prepared[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in kinds["categorical"]:
        prepared[name] = frame[name].astype("string").fillna("__missing__").astype(str)
    for name in kinds["text"]:
        prepared[name] = frame[name].astype("string").fillna("").astype(str)
    return prepared.loc[:, columns]


def build_preprocessor(kinds: dict[str, list[str]]) -> ColumnTransformer:
    """A new unfitted transformer. Call this inside each out-of-fold split."""

    transformers = []
    if kinds["numeric"]:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, kinds["numeric"]))
    if kinds["categorical"]:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(("categorical", categorical, kinds["categorical"]))
    for name in kinds["text"]:
        transformers.append(
            (
                f"text_{name}",
                TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1),
                name,
            )
        )
    if not transformers:
        raise ValueError("No feature columns to transform")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
