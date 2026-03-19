import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.prototype.embeddinggemma_lsr.cli import (
    apply_config_overrides,
    parser_default_values,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LENS model-creation pipeline: HF backbone prep followed by "
            "clustered compact-head generation."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--backbone-script",
        type=str,
        default="script/model_creation/lens/build_hf_backbone.py",
    )
    parser.add_argument(
        "--backbone-config",
        type=str,
        default="config/model_creation/lens/backbone.yaml",
    )
    parser.add_argument(
        "--cluster-script",
        type=str,
        default="script/model_creation/lens/build_clustered_head.py",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="config/model_creation/lens/cluster_head.yaml",
    )
    parser.add_argument(
        "--run-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print-commands-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--run-log-dir",
        type=str,
        default="outputs/model_creation/lens/pipeline_runs",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def _default_values() -> dict[str, Any]:
    return parser_default_values(_build_parser())


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    return apply_config_overrides(args, defaults=_default_values())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config_mapping(path: Path) -> dict[str, Any]:
    payload_raw: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload_raw, dict):
        raise ValueError(f"Expected top-level mapping in config file: {path}")
    return payload_raw


def _build_stage_command(
    *,
    python_bin: str,
    script_path: str,
    config_path: str,
) -> list[str]:
    return [python_bin, script_path, "--config", config_path]


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def _append_event(run_dir: Path, payload: dict[str, Any]) -> None:
    event_payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


def _validate_expected_path(config_path: Path, key: str) -> Path:
    config_payload: dict[str, Any] = _load_config_mapping(config_path)
    raw_value: Any | None = config_payload.get(key)
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(f"Missing required key {key!r} in config: {config_path}")
    return Path(str(raw_value))


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()
    args = _apply_config_overrides(args)

    repo_root: Path = _repo_root()
    backbone_config_path: Path = repo_root / str(args.backbone_config)
    cluster_config_path: Path = repo_root / str(args.cluster_config)
    backbone_output_dir: Path = _validate_expected_path(backbone_config_path, "output_dir")
    cluster_model_dir: Path = _validate_expected_path(cluster_config_path, "model_dir")
    cluster_output_dir: Path = _validate_expected_path(cluster_config_path, "output_dir")

    run_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    if not bool(args.print_commands_only):
        run_dir = _resolve_run_dir(repo_root, args)
        manifest = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "repo_root": str(repo_root),
            "arguments": vars(args),
            "paths": {
                "backbone_config": str(backbone_config_path),
                "cluster_config": str(cluster_config_path),
                "backbone_output_dir": str(repo_root / backbone_output_dir),
                "cluster_model_dir": str(repo_root / cluster_model_dir),
                "cluster_output_dir": str(repo_root / cluster_output_dir),
            },
            "stages": {
                "backbone": {
                    "enabled": bool(args.run_backbone),
                    "status": "pending" if bool(args.run_backbone) else "disabled",
                },
                "cluster": {
                    "enabled": bool(args.run_cluster),
                    "status": "pending" if bool(args.run_cluster) else "disabled",
                },
            },
        }
        _write_json(run_dir / "manifest.json", manifest)
        _append_event(run_dir, {"event": "pipeline_started"})

    stage_specs: list[tuple[str, bool, str, str]] = [
        (
            "backbone",
            bool(args.run_backbone),
            str(args.backbone_script),
            str(args.backbone_config),
        ),
        (
            "cluster",
            bool(args.run_cluster),
            str(args.cluster_script),
            str(args.cluster_config),
        ),
    ]
    stage_name: str
    should_run: bool
    script_path: str
    config_path: str
    for stage_name, should_run, script_path, config_path in stage_specs:
        if not should_run:
            continue
        if stage_name == "cluster" and not bool(args.print_commands_only):
            resolved_model_dir: Path = repo_root / cluster_model_dir
            if not resolved_model_dir.exists():
                raise FileNotFoundError(
                    "Cluster stage requires an existing model_dir. "
                    f"Missing: {resolved_model_dir}"
                )
        command: list[str] = _build_stage_command(
            python_bin=str(args.python_bin),
            script_path=script_path,
            config_path=config_path,
        )
        if run_dir is not None and manifest is not None:
            manifest["stages"][stage_name]["status"] = "running"
            manifest["stages"][stage_name]["command"] = command
            _write_json(run_dir / "manifest.json", manifest)
            _append_event(
                run_dir,
                {"event": "stage_started", "stage": stage_name, "command": command},
            )
        try:
            _run_stage(
                repo_root=repo_root,
                command=command,
                print_commands_only=bool(args.print_commands_only),
            )
        except Exception:
            if run_dir is not None and manifest is not None:
                manifest["stages"][stage_name]["status"] = "failed"
                manifest["failed_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(run_dir / "manifest.json", manifest)
                _append_event(
                    run_dir,
                    {"event": "stage_failed", "stage": stage_name},
                )
            raise
        if run_dir is not None and manifest is not None:
            manifest["stages"][stage_name]["status"] = "completed"
            _write_json(run_dir / "manifest.json", manifest)
            _append_event(
                run_dir,
                {"event": "stage_completed", "stage": stage_name},
            )

    if not bool(args.print_commands_only):
        print(f"Backbone artifacts: {repo_root / backbone_output_dir}")
        print(f"Clustered model dir: {repo_root / cluster_output_dir}")
    if run_dir is not None and manifest is not None:
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["status"] = "completed"
        _write_json(run_dir / "manifest.json", manifest)
        _append_event(run_dir, {"event": "pipeline_completed"})
        print(f"Pipeline run log: {run_dir}")


if __name__ == "__main__":
    main()
