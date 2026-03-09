import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.prototype.embeddinggemma_lsr.artifacts import (
    DF_MAP_FILENAME,
    VOCAB_LIST_FILENAME,
    load_vocab_artifacts,
    resolve_term_stats_cache_path,
    write_json,
)
from src.prototype.embeddinggemma_lsr.cli import (
    apply_config_overrides,
    parser_default_values,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EmbeddingGemma SPLADE model-creation pipeline: target vocab "
            "build followed by HF backbone build."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--vocab-script",
        type=str,
        default="script/model_creation/embeddinggemma_splade/build_target_vocab.py",
    )
    parser.add_argument(
        "--vocab-config",
        type=str,
        default="config/model_creation/embeddinggemma_splade/vocab.yaml",
    )
    parser.add_argument(
        "--backbone-script",
        type=str,
        default="script/model_creation/embeddinggemma_splade/build_hf_backbone.py",
    )
    parser.add_argument(
        "--backbone-config",
        type=str,
        default="config/model_creation/embeddinggemma_splade/hf_backbone.yaml",
    )
    parser.add_argument(
        "--run-vocab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the vocab build stage.",
    )
    parser.add_argument(
        "--run-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the HF backbone build stage.",
    )
    parser.add_argument(
        "--selection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the vocab stage with --selection-only against cached term statistics.",
    )
    parser.add_argument(
        "--skip-vocab-if-cache-exists",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the vocab stage when its term-statistics cache already exists.",
    )
    parser.add_argument(
        "--print-commands-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the resolved stage commands without running them.",
    )
    parser.add_argument(
        "--run-log-dir",
        type=str,
        default="outputs/model_creation/embeddinggemma_splade/pipeline_runs",
        help="Directory where per-run pipeline manifests and event logs are written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to a timestamped directory.",
    )
    return parser


def _default_values() -> dict[str, Any]:
    return parser_default_values(_build_parser())


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    return apply_config_overrides(args, defaults=_default_values())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config_mapping(path: Path) -> dict[str, Any]:
    payload: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in config file: {path}")
    return payload


def _resolve_vocab_paths(vocab_config_path: Path) -> tuple[Path, Path]:
    vocab_cfg: dict[str, Any] = _load_config_mapping(vocab_config_path)
    output_dir_raw: Any | None = vocab_cfg.get("output_dir")
    if output_dir_raw is None or not str(output_dir_raw).strip():
        raise ValueError(f"vocab config missing output_dir: {vocab_config_path}")
    output_dir: Path = Path(str(output_dir_raw))
    cache_path: Path = resolve_term_stats_cache_path(
        output_dir=output_dir,
        configured_path=vocab_cfg.get("term_stats_cache_path"),
    )
    return output_dir, cache_path


def _build_stage_command(
    *,
    python_bin: str,
    script_path: str,
    config_path: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    command: list[str] = [python_bin, script_path, "--config", config_path]
    if extra_args:
        command.extend(extra_args)
    return command


def _run_stage(*, repo_root: Path, command: list[str], print_commands_only: bool) -> None:
    printable_command: str = " ".join(shlex.quote(part) for part in command)
    print(printable_command)
    if bool(print_commands_only):
        return
    subprocess.run(command, cwd=str(repo_root), check=True)


def _resolve_run_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    base_dir: Path = repo_root / Path(str(args.run_log_dir))
    base_dir.mkdir(parents=True, exist_ok=True)
    requested_name: str | None = (
        str(args.run_name).strip() if args.run_name is not None else None
    )
    candidate_name: str = (
        requested_name
        if requested_name
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir: Path = base_dir / candidate_name
    suffix: int = 1
    while run_dir.exists():
        run_dir = base_dir / f"{candidate_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_event(run_dir: Path, payload: dict[str, Any]) -> None:
    events_path: Path = run_dir / "events.jsonl"
    event_payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    with events_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


def _write_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "manifest.json", payload, sort_keys=True)


def _validate_vocab_outputs(vocab_output_dir: Path) -> None:
    load_vocab_artifacts(vocab_output_dir)


def _validate_backbone_inputs(backbone_config_path: Path, repo_root: Path) -> None:
    backbone_cfg: dict[str, Any] = _load_config_mapping(backbone_config_path)
    vocab_dir_raw: Any | None = backbone_cfg.get("vocab_artifact_dir")
    if vocab_dir_raw is None or not str(vocab_dir_raw).strip():
        raise ValueError(f"backbone config missing vocab_artifact_dir: {backbone_config_path}")
    vocab_dir: Path = Path(str(vocab_dir_raw))
    if not vocab_dir.is_absolute():
        vocab_dir = repo_root / vocab_dir
    load_vocab_artifacts(vocab_dir)


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()
    args = _apply_config_overrides(args)

    repo_root: Path = _repo_root()
    vocab_config_path: Path = repo_root / str(args.vocab_config)
    backbone_config_path: Path = repo_root / str(args.backbone_config)
    vocab_output_dir, vocab_cache_path = _resolve_vocab_paths(vocab_config_path)
    resolved_vocab_cache_path: Path = (
        vocab_cache_path if vocab_cache_path.is_absolute() else repo_root / vocab_cache_path
    )
    backbone_cfg: dict[str, Any] = _load_config_mapping(backbone_config_path)
    backbone_output_dir_raw: Any | None = backbone_cfg.get("output_dir")
    backbone_output_dir: Path | None = None
    if backbone_output_dir_raw is not None and str(backbone_output_dir_raw).strip():
        backbone_output_dir = repo_root / Path(str(backbone_output_dir_raw))

    run_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    if not bool(args.print_commands_only):
        run_dir = _resolve_run_dir(repo_root, args)
        manifest = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "repo_root": str(repo_root),
            "arguments": vars(args),
            "paths": {
                "vocab_config": str(vocab_config_path),
                "vocab_output_dir": str(repo_root / vocab_output_dir),
                "vocab_cache_path": str(resolved_vocab_cache_path),
                "backbone_config": str(backbone_config_path),
                "backbone_output_dir": (
                    str(backbone_output_dir) if backbone_output_dir is not None else None
                ),
            },
            "stages": {
                "vocab": {
                    "enabled": bool(args.run_vocab),
                    "status": "pending" if bool(args.run_vocab) else "disabled",
                },
                "backbone": {
                    "enabled": bool(args.run_backbone),
                    "status": "pending" if bool(args.run_backbone) else "disabled",
                },
            },
        }
        _write_manifest(run_dir, manifest)
        _append_event(run_dir, {"event": "pipeline_started"})

    should_skip_vocab: bool = bool(args.skip_vocab_if_cache_exists) and vocab_cache_path.exists()
    if (
        bool(args.selection_only)
        and not bool(args.print_commands_only)
        and not vocab_cache_path.exists()
    ):
        if run_dir is not None and manifest is not None:
            manifest["failed_at"] = datetime.now().isoformat(timespec="seconds")
            manifest["error"] = (
                "selection-only requested but term statistics cache is missing: "
                f"{vocab_cache_path}"
            )
            _write_manifest(run_dir, manifest)
            _append_event(
                run_dir,
                {
                    "event": "pipeline_failed",
                    "reason": "missing_selection_cache",
                    "path": str(resolved_vocab_cache_path),
                },
            )
        raise FileNotFoundError(
            "selection-only requested but term statistics cache is missing: "
            f"{vocab_cache_path}"
        )

    if bool(args.run_vocab) and not should_skip_vocab:
        vocab_extra_args: list[str] = []
        if bool(args.selection_only):
            vocab_extra_args.append("--selection-only")
        vocab_command: list[str] = _build_stage_command(
            python_bin=str(args.python_bin),
            script_path=str(args.vocab_script),
            config_path=str(args.vocab_config),
            extra_args=vocab_extra_args,
        )
        if run_dir is not None and manifest is not None:
            manifest["stages"]["vocab"]["status"] = "running"
            manifest["stages"]["vocab"]["command"] = vocab_command
            _write_manifest(run_dir, manifest)
            _append_event(
                run_dir,
                {"event": "stage_started", "stage": "vocab", "command": vocab_command},
            )
        try:
            _run_stage(
                repo_root=repo_root,
                command=vocab_command,
                print_commands_only=bool(args.print_commands_only),
            )
        except Exception:
            if run_dir is not None and manifest is not None:
                manifest["stages"]["vocab"]["status"] = "failed"
                manifest["failed_at"] = datetime.now().isoformat(timespec="seconds")
                _write_manifest(run_dir, manifest)
                _append_event(run_dir, {"event": "stage_failed", "stage": "vocab"})
            raise
        if run_dir is not None and manifest is not None:
            manifest["stages"]["vocab"]["status"] = "completed"
            _write_manifest(run_dir, manifest)
            _append_event(run_dir, {"event": "stage_completed", "stage": "vocab"})
    elif bool(args.run_vocab):
        print(f"Skipping vocab stage because cache exists: {vocab_cache_path}")
        if run_dir is not None and manifest is not None:
            manifest["stages"]["vocab"]["status"] = "skipped"
            manifest["stages"]["vocab"]["reason"] = "cache_exists"
            _write_manifest(run_dir, manifest)
            _append_event(
                run_dir,
                {
                    "event": "stage_skipped",
                    "stage": "vocab",
                    "reason": "cache_exists",
                    "path": str(resolved_vocab_cache_path),
                },
            )

    if bool(args.run_backbone):
        if not bool(args.print_commands_only):
            _validate_vocab_outputs(repo_root / vocab_output_dir)
        backbone_command: list[str] = _build_stage_command(
            python_bin=str(args.python_bin),
            script_path=str(args.backbone_script),
            config_path=str(args.backbone_config),
        )
        if not bool(args.print_commands_only):
            _validate_backbone_inputs(backbone_config_path, repo_root)
        if run_dir is not None and manifest is not None:
            manifest["stages"]["backbone"]["status"] = "running"
            manifest["stages"]["backbone"]["command"] = backbone_command
            _write_manifest(run_dir, manifest)
            _append_event(
                run_dir,
                {
                    "event": "stage_started",
                    "stage": "backbone",
                    "command": backbone_command,
                },
            )
        try:
            _run_stage(
                repo_root=repo_root,
                command=backbone_command,
                print_commands_only=bool(args.print_commands_only),
            )
        except Exception:
            if run_dir is not None and manifest is not None:
                manifest["stages"]["backbone"]["status"] = "failed"
                manifest["failed_at"] = datetime.now().isoformat(timespec="seconds")
                _write_manifest(run_dir, manifest)
                _append_event(
                    run_dir,
                    {"event": "stage_failed", "stage": "backbone"},
                )
            raise
        if run_dir is not None and manifest is not None:
            manifest["stages"]["backbone"]["status"] = "completed"
            _write_manifest(run_dir, manifest)
            _append_event(
                run_dir,
                {"event": "stage_completed", "stage": "backbone"},
            )

    if bool(args.run_backbone) and not bool(args.print_commands_only):
        if backbone_output_dir is not None:
            print(f"Backbone artifacts: {backbone_output_dir}")
    if bool(args.run_vocab) and not bool(args.print_commands_only):
        print(f"Vocab artifacts: {repo_root / vocab_output_dir}")
        print(f"Vocab files: {repo_root / vocab_output_dir / VOCAB_LIST_FILENAME}")
        print(f"Vocab DF map: {repo_root / vocab_output_dir / DF_MAP_FILENAME}")
    if run_dir is not None and manifest is not None:
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["status"] = "completed"
        _write_manifest(run_dir, manifest)
        _append_event(run_dir, {"event": "pipeline_completed"})
        print(f"Pipeline run log: {run_dir}")


if __name__ == "__main__":
    main()
