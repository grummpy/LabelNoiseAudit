import threading
import urllib.request
from pathlib import Path

import pandas as pd

from labelnoiseaudit import HOST
from labelnoiseaudit.app import create_app
from labelnoiseaudit.export import sha256_file
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
    for _ in range(100):
        payload = client.get(f"/api/audits/{audit_id}").get_json()
        if payload["status"] != "running":
            break
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
