import threading
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from labelnoiseaudit import HOST
from labelnoiseaudit.app import create_app
from labelnoiseaudit.export import sha256_file
from labelnoiseaudit.serialize import read_json, write_json
from labelnoiseaudit.server import make_bound_server


def test_server_on_loopback_returns_200(tmp_path):
    assert HOST == "127.0.0.1"
    app = create_app(data_dir=tmp_path)
    server = make_bound_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert "LabelNoiseAudit" in body
            assert "Noise" in body
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_review_round_trip_does_not_touch_the_source_csv(tmp_path: Path):
    source = tmp_path / "plants.csv"
    frame = pd.DataFrame(
        {
            "petal": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2, 1.3, 5.4, 9.3],
            "width": [0.2, 0.3, 0.2, 1.5, 1.6, 1.4, 2.4, 2.5, 2.6, 0.25, 1.55, 2.45],
            "species": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a", "b", "c"],
        }
    )
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    digest = sha256_file(source)

    app = create_app(data_dir=tmp_path / "appdata")
    client = app.test_client()
    opened = client.post("/api/datasets/csv", json={"path": str(source)})
    assert opened.status_code == 200, opened.get_json()
    dataset = opened.get_json()
    started = client.post(
        "/api/audits",
        json={
            "dataset_id": dataset["dataset_id"],
            "label_column": "species",
            "feature_columns": ["petal", "width"],
            "n_splits": 3,
            "seed": 1,
        },
    )
    assert started.status_code == 200, started.get_json()
    audit_id = started.get_json()["audit_id"]
    payload = None
    for _ in range(2_000):
        payload = client.get(f"/api/audits/{audit_id}").get_json()
        if payload["status"] != "running":
            break
        time.sleep(0.01)
    assert payload["status"] == "done", payload
    row = payload["result"]["rows"][0]
    decision = client.post(
        f"/api/audits/{audit_id}/decisions",
        json={"source_index": row["source_index"], "action": "relabel", "new_label": row["predicted_label"]},
    )
    assert decision.status_code == 200
    exported = client.post(f"/api/audits/{audit_id}/export")
    assert exported.status_code == 200, exported.get_json()
    body = exported.get_json()
    cleaned = client.get(body["cleaned_url"])
    log = client.get(body["log_url"])
    assert cleaned.status_code == 200
    assert log.status_code == 200
    assert b"species" in cleaned.data
    assert source.read_bytes() == before
    assert sha256_file(source) == digest
    text = cleaned.data.decode("utf-8")
    assert "__lna_source_index" not in text


def test_image_audit_binds_its_label_even_if_request_names_path(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "appdata")
    client = app.test_client()
    dataset = client.post("/api/datasets/builtin", json={"name": "shapes"}).get_json()
    started = client.post(
        "/api/audits",
        json={"dataset_id": dataset["dataset_id"], "label_column": "path", "n_splits": 2},
    )
    assert started.status_code == 200, started.get_json()
    audit_id = started.get_json()["audit_id"]
    for _ in range(2_000):
        payload = client.get(f"/api/audits/{audit_id}").get_json()
        if payload["status"] != "running":
            break
        time.sleep(0.01)
    assert payload["status"] == "done", payload
    row = payload["result"]["rows"][0]
    source_frame = pd.read_csv(tmp_path / "appdata" / "datasets" / dataset["dataset_id"] / "source.csv")
    original = source_frame.loc[source_frame["__lna_source_index"] == row["source_index"]].iloc[0]
    changed = client.post(
        f"/api/audits/{audit_id}/decisions",
        json={"source_index": row["source_index"], "action": "relabel", "new_label": "changed-label"},
    )
    assert changed.status_code == 200, changed.get_json()
    exported = client.post(f"/api/audits/{audit_id}/export")
    assert exported.status_code == 200, exported.get_json()
    cleaned = client.get(exported.get_json()["cleaned_url"])
    frame = pd.read_csv(BytesIO(cleaned.data))
    exported_row = frame.loc[frame["path"] == original["path"]].iloc[0]
    assert exported_row["path"] == original["path"]
    assert exported_row["label"] == "changed-label"


@pytest.mark.parametrize("legacy_label", ["path", None])
def test_export_normalizes_legacy_image_audit_path_binding(tmp_path: Path, legacy_label: str | None):
    app = create_app(data_dir=tmp_path / "appdata")
    client = app.test_client()
    dataset = client.post("/api/datasets/builtin", json={"name": "shapes"}).get_json()
    started = client.post("/api/audits", json={"dataset_id": dataset["dataset_id"], "n_splits": 2})
    assert started.status_code == 200, started.get_json()
    audit_id = started.get_json()["audit_id"]
    for _ in range(2_000):
        payload = client.get(f"/api/audits/{audit_id}").get_json()
        if payload["status"] != "running":
            break
        time.sleep(0.01)
    assert payload["status"] == "done", payload
    row = payload["result"]["rows"][0]
    appdata = tmp_path / "appdata"
    request_path = appdata / "audits" / audit_id / "request.json"
    legacy_request = read_json(request_path)
    if legacy_label is None:
        legacy_request.pop("label_column")
    else:
        legacy_request["label_column"] = legacy_label
    write_json(request_path, legacy_request)
    changed = client.post(
        f"/api/audits/{audit_id}/decisions",
        json={"source_index": row["source_index"], "action": "relabel", "new_label": "changed-label"},
    )
    assert changed.status_code == 200, changed.get_json()
    exported = client.post(f"/api/audits/{audit_id}/export")
    assert exported.status_code == 200, exported.get_json()
    body = exported.get_json()
    cleaned = pd.read_csv(BytesIO(client.get(body["cleaned_url"]).data))
    original_frame = pd.read_csv(appdata / "datasets" / dataset["dataset_id"] / "source.csv")
    original = original_frame.loc[original_frame["__lna_source_index"] == row["source_index"]].iloc[0]
    exported_row = cleaned.loc[cleaned["path"] == original["path"]].iloc[0]
    assert exported_row["path"] == original["path"]
    assert exported_row["label"] == "changed-label"
    log = client.get(body["log_url"]).get_json()
    assert log["audit"]["label_binding"] == {
        "status": "normalized_legacy_image_audit",
        "stored_label_column": legacy_label or "",
        "effective_label_column": "label",
    }
