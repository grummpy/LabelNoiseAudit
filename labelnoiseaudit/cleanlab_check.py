"""Optional cleanlab cross-check.

Imported only when an audit asks for the comparison. A missing install
returns ``available: False`` and the rest of the audit still finishes.
"""

from __future__ import annotations

import numpy as np


def cleanlab_crosscheck(labels: np.ndarray, proba: np.ndarray) -> dict[str, object]:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        return {"available": False, "error": None, "flagged_indices": []}
    try:
        ranked = find_label_issues(
            labels=np.asarray(labels),
            pred_probs=np.asarray(proba, dtype=float),
            return_indices_ranked_by="self_confidence",
        )
    except Exception as exc:
        return {"available": True, "error": str(exc), "flagged_indices": []}
    indices = [int(index) for index in np.asarray(ranked).tolist()]
    return {"available": True, "error": None, "flagged_indices": indices}
