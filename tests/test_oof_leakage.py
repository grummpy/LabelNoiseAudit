import numpy as np
from sklearn.model_selection import StratifiedKFold

from labelnoiseaudit.audit import audit_matrix, result_to_dict
from labelnoiseaudit.oof import choose_n_splits


class Spy:
    seen: list[set[int]] = []

    def fit(self, features, y=None):
        Spy.seen.append(set(np.asarray(features)[:, 0].astype(int).tolist()))
        return self

    def transform(self, features):
        return np.asarray(features, dtype=float)[:, 1:]


def test_preprocessor_is_fit_on_the_train_fold_only():
    Spy.seen = []
    rng = np.random.default_rng(0)
    count = 20
    features = np.column_stack(
        [
            np.arange(count * 2),
            np.vstack(
                [
                    rng.normal(0.0, 0.2, size=(count, 3)),
                    rng.normal(3.0, 0.2, size=(count, 3)),
                ]
            ),
        ]
    )
    labels = np.array([0] * count + [1] * count)
    seed = 0
    n_splits = 5
    audit_matrix(
        features,
        labels,
        seed=seed,
        n_splits=n_splits,
        preprocessor_factory=Spy,
        use_cleanlab=False,
    )
    assert len(Spy.seen) == n_splits
    folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for (train_idx, test_idx), seen in zip(folder.split(np.zeros(len(labels)), labels), Spy.seen, strict=True):
        train_ids = set(features[train_idx, 0].astype(int).tolist())
        test_ids = set(features[test_idx, 0].astype(int).tolist())
        assert seen == train_ids
        assert seen.isdisjoint(test_ids)


def test_neighbors_do_not_include_the_row_itself():
    rng = np.random.default_rng(2)
    features = np.vstack(
        [
            rng.normal(0.0, 0.4, size=(30, 3)),
            rng.normal(4.0, 0.4, size=(30, 3)),
        ]
    )
    labels = np.array([0] * 30 + [1] * 30)
    result = audit_matrix(features, labels, seed=2, n_neighbors=5, use_cleanlab=False)
    for row in result.rows_by_source:
        assert row.source_index not in row.neighbor_indices
        assert len(row.neighbor_indices) == 5


def test_split_count_shrinks_for_a_small_class():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert choose_n_splits(labels, 5) == 3


def test_every_exported_row_has_one_recorded_oof_fold_and_class_order():
    features = np.array([[0.0], [0.1], [0.2], [0.3], [3.0], [3.1], [3.2], [3.3]])
    labels = np.array(["left"] * 4 + ["right"] * 4)
    payload = result_to_dict(audit_matrix(features, labels, n_splits=2, seed=9, use_cleanlab=False))
    assert payload["provenance"]["configuration"]["class_order"] == ["left", "right"]
    assert {row["oof_fold"] for row in payload["rows"]} == {1, 2}
    for row in payload["rows"]:
        assert set(row["oof_probabilities"]) == {"left", "right"}
