import builtins

import pytest

from labelnoiseaudit.cleanlab_check import cleanlab_crosscheck
from labelnoiseaudit.embeddings import embed_paths


def test_embeddings_refuse_without_opt_in():
    with pytest.raises(RuntimeError, match="opt in"):
        embed_paths(["/tmp/does-not-matter.png"], allow_download=False)


def test_embedding_request_without_opt_in_never_imports_optional_packages(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torchvision"):
            raise AssertionError("optional embedding package was imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="opt in"):
        embed_paths(["/tmp/does-not-matter.png"], allow_download=False)


def test_embedding_missing_optional_packages_has_a_clear_refusal(monkeypatch):
    original_import = builtins.__import__

    def missing_optional(name, *args, **kwargs):
        if name == "torch" or name.startswith("torchvision"):
            raise ImportError("synthetic missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_optional)
    with pytest.raises(RuntimeError, match="extra install"):
        embed_paths(["/tmp/does-not-matter.png"], allow_download=True)


def test_cleanlab_is_optional():
    import numpy as np

    result = cleanlab_crosscheck(np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]))
    assert "available" in result
    assert "flagged_indices" in result
    if not result["available"]:
        assert result["flagged_indices"] == []
