"""LabelNoiseAudit ranks rows that are likely mislabeled.

Thread limits are set before NumPy is imported so out-of-fold fits stay
on one thread and repeated runs with the same seed match.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

__version__ = "0.1.0"

DEFAULT_SEED = 42
DEFAULT_SPLITS = 5
DEFAULT_NEIGHBORS = 5
DEFAULT_PORT = 8741
HOST = "127.0.0.1"
