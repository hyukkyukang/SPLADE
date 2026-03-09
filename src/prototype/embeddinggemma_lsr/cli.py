import argparse
from typing import Any

import torch
from omegaconf import OmegaConf


def parser_default_values(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    action: argparse.Action
    for action in parser._actions:
        if action.dest in {None, "help"}:
            continue
        defaults[str(action.dest)] = action.default
    return defaults


def apply_config_overrides(
    args: argparse.Namespace,
    *,
    defaults: dict[str, Any] | None = None,
) -> argparse.Namespace:
    config_path: str | None = getattr(args, "config", None)
    if config_path is None:
        return args

    payload_raw: Any = OmegaConf.to_container(
        OmegaConf.load(config_path),
        resolve=True,
    )
    if payload_raw is None:
        return args
    if not isinstance(payload_raw, dict):
        raise ValueError(
            f"Expected top-level mapping in config override file: {config_path}"
        )

    effective_defaults: dict[str, Any] = defaults or {}
    key: str
    value: Any
    for key, value in payload_raw.items():
        if not hasattr(args, key):
            continue
        if key in effective_defaults and getattr(args, key) != effective_defaults[key]:
            continue
        setattr(args, key, value)
    return args


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized: str = str(dtype_name).strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_torch_device(device_value: str) -> torch.device:
    normalized: str = str(device_value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)
