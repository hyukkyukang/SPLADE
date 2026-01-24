import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_tsv(path: str, has_header: bool = False) -> list[list[str]]:
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if has_header and rows:
        rows = rows[1:]
    return rows


def write_jsonl(path: str, items: Iterable[dict[str, Any]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
