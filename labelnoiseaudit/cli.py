"""Command line: serve the review UI, audit a file, benchmark, or export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from labelnoiseaudit import __version__
from labelnoiseaudit.audit import audit_image_records, audit_table, result_to_dict
from labelnoiseaudit.benchmark import markdown_table, run_benchmark
from labelnoiseaudit.export import OverwriteRefused, export_review
from labelnoiseaudit.images import iter_class_folder, read_image_csv
from labelnoiseaudit.serialize import read_json, write_json
from labelnoiseaudit.server import serve


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in {"serve", "audit", "benchmark", "export", "-h", "--help"}:
        argv = ["serve", *argv]

    parser = argparse.ArgumentParser(
        prog="labelnoiseaudit",
        description="Rank dataset rows that are most likely mislabeled.",
    )
    parser.add_argument("--version", action="version", version=f"labelnoiseaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="Open the local review UI")
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--no-browser", action="store_true")
    serve_parser.add_argument("--data-dir", type=Path, default=None)

    audit_parser = sub.add_parser("audit", help="Score a CSV or an image folder")
    audit_parser.add_argument("--csv", type=Path)
    audit_parser.add_argument("--images", type=Path)
    audit_parser.add_argument("--label")
    audit_parser.add_argument("--features", nargs="*")
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.add_argument("--folds", type=int, default=5)
    audit_parser.add_argument("--neighbors", type=int, default=5)
    audit_parser.add_argument("--out", type=Path, required=True)
    audit_parser.add_argument("--embeddings", action="store_true")
    audit_parser.add_argument("--allow-weight-download", action="store_true")

    bench_parser = sub.add_parser("benchmark", help="Score synthetic label noise")
    bench_parser.add_argument("--seed", type=int, default=42)
    bench_parser.add_argument("--output", type=Path)
    bench_parser.add_argument("--markdown", type=Path)

    export_parser = sub.add_parser("export", help="Apply review decisions to a new CSV")
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--label", required=True)
    export_parser.add_argument("--decisions", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.add_argument("--log", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir)
        return 0
    if args.command == "audit":
        return _audit(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "export":
        return _export(args)
    parser.error(f"Unknown command {args.command}")
    return 2


def _refuse_same_path(source: Path, dest: Path) -> None:
    if source.resolve() == dest.resolve():
        raise SystemExit(
            "Refusing to overwrite the input. Choose a different output path."
        )


def _audit(args) -> int:
    if bool(args.csv) == bool(args.images):
        raise SystemExit("Pass exactly one of --csv or --images.")
    if args.csv is not None:
        _refuse_same_path(args.csv, args.out)
        frame = pd.read_csv(args.csv)
        label = args.label
        if not label:
            raise SystemExit("--label is required for a CSV audit.")
        print(f"Auditing {len(frame)} rows from {args.csv}")
        result = audit_table(
            frame,
            label,
            feature_columns=args.features,
            n_splits=args.folds,
            seed=args.seed,
            n_neighbors=args.neighbors,
            progress=print,
        )
    else:
        path = args.images
        if path.is_dir():
            records = iter_class_folder(path)
        else:
            records = read_image_csv(path)
        if args.out.resolve() == path.resolve():
            raise SystemExit("Refusing to overwrite the input. Choose a different output path.")
        print(f"Auditing {len(records)} images from {path}")
        result = audit_image_records(
            records,
            n_splits=args.folds,
            seed=args.seed,
            n_neighbors=args.neighbors,
            embeddings=args.embeddings,
            allow_download=args.allow_weight_download,
            progress=print,
        )
    payload = result_to_dict(result)
    write_json(args.out, payload)
    flagged = payload["summary"]["n_flagged"]
    print(f"Wrote {args.out} ({flagged} rows marked Check).")
    return 0


def _benchmark(args) -> int:
    report = run_benchmark(args.seed, progress=print)
    text = markdown_table(report)
    print(text)
    if args.output is not None:
        write_json(args.output, report)
        print(f"Wrote {args.output}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
        print(f"Wrote {args.markdown}")
    return 0


def _export(args) -> int:
    _refuse_same_path(args.source, args.out)
    _refuse_same_path(args.source, args.log)
    frame = pd.read_csv(args.source)
    decisions = read_json(args.decisions)
    if isinstance(decisions, dict) and "rows" in decisions and isinstance(decisions["rows"], list):
        mapped = {}
        for row in decisions["rows"]:
            mapped[str(row["source_index"])] = {
                key: value for key, value in row.items() if key != "source_index"
            }
        decisions = mapped
    try:
        export_review(
            frame,
            args.label,
            decisions,
            args.out,
            args.log,
            original_path=args.source,
        )
    except OverwriteRefused as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {args.out}")
    print(f"Wrote {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
