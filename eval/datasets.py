from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser()
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        return _load_json_dataset(dataset_path)
    if suffix == ".csv":
        return _load_csv_dataset(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def _load_json_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return [item for item in payload["examples"] if isinstance(item, dict)]
    raise ValueError("JSON dataset must be a list of objects or {'examples':[...]} format.")


def _load_csv_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows
