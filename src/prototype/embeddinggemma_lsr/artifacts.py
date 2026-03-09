import json
from pathlib import Path
from typing import Any

VOCAB_LIST_FILENAME: str = "v_target.txt"
DF_MAP_FILENAME: str = "df_map.json"
VOCAB_STATS_FILENAME: str = "vocab_stats.json"
VOCAB_MANIFEST_FILENAME: str = "manifest.json"
TERM_STATS_CACHE_FILENAME: str = "term_statistics.pkl"


def read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_vocab_artifacts(vocab_artifact_dir: Path) -> tuple[list[str], dict[str, int]]:
    vocab_path: Path = vocab_artifact_dir / VOCAB_LIST_FILENAME
    df_map_path: Path = vocab_artifact_dir / DF_MAP_FILENAME

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing file: {vocab_path}")
    if not df_map_path.is_file():
        raise FileNotFoundError(f"Missing file: {df_map_path}")

    target_vocab: list[str] = read_nonempty_lines(vocab_path)
    df_map_raw: dict[str, Any] = json.loads(df_map_path.read_text(encoding="utf-8"))
    df_map: dict[str, int] = {str(key): int(value) for key, value in df_map_raw.items()}
    return target_vocab, df_map


def resolve_term_stats_cache_path(
    *,
    output_dir: Path,
    configured_path: str | None,
) -> Path:
    if configured_path is not None and str(configured_path).strip():
        return Path(str(configured_path))
    return output_dir / TERM_STATS_CACHE_FILENAME


def write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys),
        encoding="utf-8",
    )


def write_text_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
