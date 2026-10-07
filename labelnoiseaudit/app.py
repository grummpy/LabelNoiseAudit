"""Local review server. It binds only to 127.0.0.1 from ``server.py``."""

from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from labelnoiseaudit import __version__
from labelnoiseaudit.audit import audit_image_records, audit_table, result_to_dict
from labelnoiseaudit.datasets import load_builtin_frame, materialize_shapes
from labelnoiseaudit.embeddings import embeddings_status
from labelnoiseaudit.export import SOURCE_INDEX, OverwriteRefused, export_review
from labelnoiseaudit.features import (
    default_feature_columns,
    infer_column_kinds,
    infer_kind,
    suggest_label_column,
)
from labelnoiseaudit.images import (
    glyph_png,
    iter_class_folder,
    read_image_csv,
    safe_extract_zip,
    thumbnail_png,
)
from labelnoiseaudit.serialize import read_json, to_builtin, write_json

STAMP = re.compile(r"^\d{8}-\d{6}(?:-\d+)?$")
EXPORT_FILES = {"cleaned.csv", "audit-log.json"}


def web_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def create_app(data_dir: Path | None = None) -> Flask:
    folder = Path(data_dir) if data_dir is not None else Path.cwd() / "data"
    folder.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=str(web_dir()), static_url_path="/static")
    app.config["DATA_DIR"] = folder
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    # Local UI files change with the install. Do not let the browser keep a stale app.js.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.get("/")
    def index():
        return send_from_directory(web_dir(), "index.html")

    @app.get("/favicon.ico")
    def favicon():
        icon = web_dir() / "favicon.ico"
        if not icon.is_file():
            abort(404)
        return send_from_directory(web_dir(), "favicon.ico")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "version": __version__, "host": "127.0.0.1"})

    @app.get("/api/meta")
    def meta():
        return jsonify(
            {
                "version": __version__,
                "builtins": [
                    {"id": "iris", "title": "Iris", "detail": "150 rows, 4 measurements, 3 species"},
                    {"id": "wine", "title": "Wine", "detail": "178 rows, 13 measurements, 3 cultivars"},
                    {"id": "digits", "title": "Digits", "detail": "1,797 handwritten digits, 8×8"},
                    {"id": "field_notes", "title": "Field notes", "detail": "Generated notes in four topics"},
                    {"id": "shapes", "title": "Shapes", "detail": "Generated sun and tile images"},
                ],
                "embeddings": embeddings_status(),
            }
        )

    @app.post("/api/datasets/builtin")
    def open_builtin():
        body = request.get_json(force=True, silent=True) or {}
        name = str(body.get("name", "")).strip().lower()
        try:
            if name == "shapes":
                dest = folder / "builtins" / "shapes"
                materialize_shapes(dest)
                meta_payload = _store_images(folder, iter_class_folder(dest), original_path=dest, builtin="shapes")
            else:
                frame, label, thumbnail = load_builtin_frame(name)
                meta_payload = _store_table(
                    folder,
                    frame,
                    thumbnail=thumbnail,
                    original_path=None,
                    builtin=name,
                    suggested_label=label,
                )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(meta_payload)

    @app.post("/api/datasets/csv")
    def open_csv():
        try:
            if "file" in request.files:
                upload = request.files["file"]
                filename = secure_filename(upload.filename or "upload.csv") or "upload.csv"
                target_dir = folder / "uploads" / uuid.uuid4().hex[:12]
                target_dir.mkdir(parents=True)
                target = target_dir / filename
                upload.save(target)
                frame = pd.read_csv(target)
                original = target
            else:
                body = request.get_json(force=True, silent=True) or {}
                raw_path = str(body.get("path", "")).strip()
                if not raw_path:
                    raise ValueError("Choose a CSV file or a path on this computer.")
                original = Path(raw_path).expanduser().resolve()
                if not original.is_file():
                    raise FileNotFoundError(f"CSV not found: {original}")
                frame = pd.read_csv(original)
            if frame.empty:
                raise ValueError("The CSV has no rows.")
            payload = _store_table(
                folder,
                frame,
                thumbnail="none",
                original_path=original,
                builtin=None,
                suggested_label=suggest_label_column(frame),
            )
        except (ValueError, FileNotFoundError, pd.errors.ParserError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.post("/api/datasets/images")
    def open_images():
        try:
            if "file" in request.files:
                upload = request.files["file"]
                filename = secure_filename(upload.filename or "images.zip") or "images.zip"
                target_dir = folder / "uploads" / uuid.uuid4().hex[:12]
                target_dir.mkdir(parents=True)
                zip_path = target_dir / filename
                upload.save(zip_path)
                extracted = safe_extract_zip(zip_path, target_dir / "unzipped")
                records = _records_from_image_root(extracted)
                original = extracted
            else:
                body = request.get_json(force=True, silent=True) or {}
                raw_path = str(body.get("path", "")).strip()
                if not raw_path:
                    raise ValueError("Choose an image folder, a CSV of paths, or a zip.")
                original = Path(raw_path).expanduser().resolve()
                if not original.exists():
                    raise FileNotFoundError(f"Path not found: {original}")
                records = _records_from_image_root(original)
            payload = _store_images(folder, records, original_path=original, builtin=None)
        except (ValueError, FileNotFoundError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.get("/api/datasets/<dataset_id>")
    def get_dataset(dataset_id: str):
        meta_path = _dataset_dir(folder, dataset_id) / "meta.json"
        if not meta_path.is_file():
            abort(404)
        return jsonify(read_json(meta_path))

    @app.post("/api/audits")
    def start_audit():
        body = request.get_json(force=True, silent=True) or {}
        dataset_id = str(body.get("dataset_id", "")).strip()
        try:
            dataset = _dataset_dir(folder, dataset_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not (dataset / "meta.json").is_file():
            return jsonify({"error": "Dataset not found."}), 404
        audit_id = uuid.uuid4().hex[:12]
        audit_dir = folder / "audits" / audit_id
        audit_dir.mkdir(parents=True)
        write_json(audit_dir / "decisions.json", {})
        write_json(audit_dir / "status.json", {"status": "running", "progress": "Starting"})
        params = {
            "dataset_id": dataset_id,
            "label_column": body.get("label_column"),
            "feature_columns": body.get("feature_columns"),
            "column_kinds": body.get("column_kinds"),
            "n_splits": int(body.get("n_splits") or 5),
            "seed": int(body.get("seed") if body.get("seed") is not None else 42),
            "n_neighbors": int(body.get("n_neighbors") or 5),
            "embeddings": bool(body.get("embeddings")),
            "allow_download": bool(body.get("allow_download")),
        }
        write_json(audit_dir / "request.json", params)
        thread = threading.Thread(
            target=_run_audit_thread,
            args=(folder, audit_id),
            daemon=True,
        )
        thread.start()
        return jsonify({"audit_id": audit_id, "status": "running"})

    @app.get("/api/audits/<audit_id>")
    def get_audit(audit_id: str):
        try:
            audit_dir = _audit_dir(folder, audit_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        status_path = audit_dir / "status.json"
        if not status_path.is_file():
            abort(404)
        payload = read_json(status_path)
        if payload.get("status") == "done" and (audit_dir / "result.json").is_file():
            payload["result"] = read_json(audit_dir / "result.json")
            payload["decisions"] = read_json(audit_dir / "decisions.json")
        return jsonify(payload)

    @app.get("/api/audits/<audit_id>/thumb/<int:source_index>")
    def thumb(audit_id: str, source_index: int):
        try:
            audit_dir = _audit_dir(folder, audit_id)
        except ValueError:
            abort(404)
        request_meta = read_json(audit_dir / "request.json")
        dataset_meta = read_json(_dataset_dir(folder, request_meta["dataset_id"]) / "meta.json")
        frame = _source_frame(folder, request_meta["dataset_id"])
        matched = frame.loc[frame[SOURCE_INDEX] == source_index]
        if matched.empty:
            abort(404)
        row = matched.iloc[0]
        mode = dataset_meta.get("thumbnail")
        if mode == "images":
            png = thumbnail_png(Path(str(row["path"])))
        elif mode == "glyphs":
            pixels = [row[f"pixel_{index}"] for index in range(64)]
            png = glyph_png(pixels)
        else:
            abort(404)
        from io import BytesIO

        return send_file(BytesIO(png), mimetype="image/png")

    @app.post("/api/audits/<audit_id>/decisions")
    def update_decision(audit_id: str):
        try:
            audit_dir = _audit_dir(folder, audit_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not (audit_dir / "decisions.json").is_file():
            abort(404)
        body = request.get_json(force=True, silent=True) or {}
        action = str(body.get("action", "keep"))
        if action not in {"keep", "relabel", "drop"}:
            return jsonify({"error": "Action must be keep, relabel, or drop."}), 400
        try:
            source_index = int(body["source_index"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "source_index is required."}), 400
        decisions = read_json(audit_dir / "decisions.json")
        entry = {"action": action}
        if action == "relabel":
            new_label = str(body.get("new_label", "")).strip()
            if not new_label:
                return jsonify({"error": "Relabel needs a new label."}), 400
            entry["new_label"] = new_label
        decisions[str(source_index)] = entry
        write_json(audit_dir / "decisions.json", decisions)
        return jsonify({"decisions": decisions})

    @app.post("/api/audits/<audit_id>/export")
    def export_audit(audit_id: str):
        try:
            audit_dir = _audit_dir(folder, audit_id)
            request_meta = read_json(audit_dir / "request.json")
            dataset_meta = read_json(_dataset_dir(folder, request_meta["dataset_id"]) / "meta.json")
            frame = _source_frame(folder, request_meta["dataset_id"])
            decisions = read_json(audit_dir / "decisions.json")
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            export_dir = audit_dir / "exports" / stamp
            suffix = 2
            while export_dir.exists():
                export_dir = audit_dir / "exports" / f"{stamp}-{suffix}"
                suffix += 1
            original = dataset_meta.get("original_path")
            result = read_json(audit_dir / "result.json") if (audit_dir / "result.json").is_file() else {}
            log = export_review(
                frame,
                request_meta["label_column"],
                decisions,
                export_dir / "cleaned.csv",
                export_dir / "audit-log.json",
                original_path=Path(original) if original else None,
                audit_meta={
                    "audit_id": audit_id,
                    "dataset_id": request_meta["dataset_id"],
                    "summary": result.get("summary", {}),
                    "noise_rates": result.get("noise_rates", {}),
                },
            )
        except OverwriteRefused as exc:
            return jsonify({"error": str(exc)}), 400
        except (ValueError, FileNotFoundError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        stamp_name = export_dir.name
        return jsonify(
            {
                "stamp": stamp_name,
                "cleaned_url": f"/api/audits/{audit_id}/exports/{stamp_name}/cleaned.csv",
                "log_url": f"/api/audits/{audit_id}/exports/{stamp_name}/audit-log.json",
                "n_out": log["n_out"],
                "n_dropped": log["n_dropped"],
                "n_relabeled": log["n_relabeled"],
            }
        )

    @app.get("/api/audits/<audit_id>/exports/<stamp>/<filename>")
    def download_export(audit_id: str, stamp: str, filename: str):
        if not STAMP.match(stamp) or filename not in EXPORT_FILES:
            abort(404)
        try:
            audit_dir = _audit_dir(folder, audit_id)
        except ValueError:
            abort(404)
        path = (audit_dir / "exports" / stamp / filename).resolve()
        exports_root = (audit_dir / "exports").resolve()
        if exports_root not in path.parents or not path.is_file():
            abort(404)
        return send_file(path, as_attachment=True, download_name=filename)

    return app


def _dataset_dir(data_dir: Path, dataset_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", dataset_id or ""):
        raise ValueError("Unknown dataset id.")
    return data_dir / "datasets" / dataset_id


def _audit_dir(data_dir: Path, audit_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", audit_id or ""):
        raise ValueError("Unknown audit id.")
    return data_dir / "audits" / audit_id


def _source_frame(data_dir: Path, dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(_dataset_dir(data_dir, dataset_id) / "source.csv")


def _store_table(data_dir, frame, *, thumbnail, original_path, builtin, suggested_label) -> dict:
    dataset_id = uuid.uuid4().hex[:12]
    dest = data_dir / "datasets" / dataset_id
    dest.mkdir(parents=True)
    stored = frame.copy()
    stored.insert(0, SOURCE_INDEX, range(len(stored)))
    stored.to_csv(dest / "source.csv", index=False)
    columns = []
    for name in frame.columns:
        series = frame[name]
        columns.append(
            {
                "name": str(name),
                "kind": infer_kind(series),
                "sample": ["" if pd.isna(value) else str(value) for value in series.head(3).tolist()],
            }
        )
    preview = to_builtin(frame.head(6).astype(object).where(frame.head(6).notna(), None).to_dict(orient="records"))
    meta = {
        "dataset_id": dataset_id,
        "kind": "table",
        "thumbnail": thumbnail,
        "builtin": builtin,
        "original_path": str(original_path) if original_path else None,
        "n_rows": int(len(frame)),
        "suggested_label": suggested_label,
        "suggested_features": default_feature_columns(frame, suggested_label),
        "columns": columns,
        "preview": preview,
        "class_sample": _class_sample(frame[suggested_label]),
    }
    write_json(dest / "meta.json", meta)
    return meta


def _store_images(data_dir, records, *, original_path, builtin) -> dict:
    frame = pd.DataFrame(
        [{"path": path, "label": label} for path, label in records]
    )
    meta = _store_table(
        data_dir,
        frame,
        thumbnail="images",
        original_path=original_path,
        builtin=builtin,
        suggested_label="label",
    )
    meta["kind"] = "images"
    meta["suggested_features"] = []
    write_json(data_dir / "datasets" / meta["dataset_id"] / "meta.json", meta)
    return meta


def _class_sample(series: pd.Series) -> list[str]:
    values = [str(value) for value in series.dropna().unique().tolist()]
    values.sort()
    return values[:12]


def _records_from_image_root(path: Path) -> list[tuple[str, str]]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return read_image_csv(path)
    if path.is_dir():
        csvs = [item for item in path.iterdir() if item.suffix.lower() == ".csv" and item.is_file()]
        subdirs = [item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")]
        if subdirs:
            return iter_class_folder(path)
        if len(csvs) == 1:
            return read_image_csv(csvs[0])
    raise ValueError("Use a class-per-folder directory, a path/label CSV, or a zip of either.")


def _run_audit_thread(data_dir: Path, audit_id: str) -> None:
    audit_dir = data_dir / "audits" / audit_id
    status_path = audit_dir / "status.json"

    def progress(message: str) -> None:
        write_json(status_path, {"status": "running", "progress": message})

    try:
        params = read_json(audit_dir / "request.json")
        dataset_meta = read_json(_dataset_dir(data_dir, params["dataset_id"]) / "meta.json")
        frame = _source_frame(data_dir, params["dataset_id"])
        feature_frame = frame.drop(columns=[SOURCE_INDEX])
        if dataset_meta["kind"] == "images":
            records = list(zip(feature_frame["path"].astype(str), feature_frame["label"].astype(str), strict=True))
            result = audit_image_records(
                records,
                n_splits=params["n_splits"],
                seed=params["seed"],
                n_neighbors=params["n_neighbors"],
                embeddings=params["embeddings"],
                allow_download=params["allow_download"],
                progress=progress,
            )
        else:
            label = str(params["label_column"] or dataset_meta["suggested_label"])
            features = params["feature_columns"] or default_feature_columns(feature_frame, label)
            kinds = params["column_kinds"]
            if kinds is None:
                kinds = infer_column_kinds(feature_frame, list(features))
            result = audit_table(
                feature_frame,
                label,
                feature_columns=list(features),
                column_kinds=kinds,
                n_splits=params["n_splits"],
                seed=params["seed"],
                n_neighbors=params["n_neighbors"],
                progress=progress,
            )
        payload = result_to_dict(result)
        payload["thumbnail"] = dataset_meta.get("thumbnail", "none")
        payload["dataset_id"] = params["dataset_id"]
        payload["label_column"] = params["label_column"] or dataset_meta["suggested_label"]
        write_json(audit_dir / "result.json", payload)
        write_json(status_path, {"status": "done", "progress": "Finished"})
    except Exception as exc:
        write_json(status_path, {"status": "error", "progress": str(exc), "error": str(exc)})
