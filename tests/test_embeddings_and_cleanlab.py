import pytest

from labelnoiseaudit.cleanlab_check import cleanlab_crosscheck
from labelnoiseaudit.embeddings import embed_paths


def test_embeddings_refuse_without_opt_in():
    with pytest.raises(RuntimeError, match="opt in"):
        embed_paths(["/tmp/does-not-matter.png"], allow_download=False)


def test_cleanlab_is_optional():
    import numpy as np

    result = cleanlab_crosscheck(np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]))
    assert "available" in result
    assert "flagged_indices" in result
    if not result["available"]:
        assert result["flagged_indices"] == []
