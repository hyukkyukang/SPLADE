from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from omegaconf import DictConfig

DDP_TIMEOUT_HOURS: int = 1


def _resolve_static_graph(cfg_section: DictConfig) -> bool:
    """Disable static_graph when gradient accumulation uses no_sync."""
    grad_accumulation: int = int(getattr(cfg_section, "grad_accumulation", 1))
    # Some PyTorch versions assert in DDP when no_sync is used with static graphs.
    return grad_accumulation <= 1


def get_cpu_trainer_kwargs(cfg_section: DictConfig) -> dict[str, Any]:
    """Build trainer kwargs for CPU execution."""
    strategy_name: str = str(cfg_section.strategy)
    num_devices: int = (
        1 if cfg_section.num_devices is None else int(cfg_section.num_devices)
    )
    kwargs: dict[str, Any] = {"accelerator": "cpu", "devices": num_devices}

    if strategy_name == "ddp":
        if num_devices > 1:
            use_static_graph: bool = _resolve_static_graph(cfg_section)
            kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
                static_graph=use_static_graph,
            )
        else:
            kwargs["strategy"] = "auto"
    elif strategy_name == "single":
        kwargs["devices"] = 1
        kwargs["strategy"] = "auto"
    else:
        raise ValueError(f"Invalid CPU strategy: {strategy_name}")

    return kwargs


def get_gpu_trainer_kwargs(cfg_section: DictConfig) -> dict[str, Any]:
    """Build trainer kwargs for CUDA execution."""
    strategy_name: str = str(cfg_section.strategy)
    detected_devices: int = int(torch.cuda.device_count())
    num_devices: int = detected_devices
    if cfg_section.num_devices is not None:
        num_devices = min(int(cfg_section.num_devices), detected_devices)

    kwargs: dict[str, Any] = {"accelerator": "cuda", "devices": num_devices}

    if strategy_name == "ddp":
        use_static_graph: bool = _resolve_static_graph(cfg_section)
        kwargs["strategy"] = DDPStrategy(
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
            static_graph=use_static_graph,
            gradient_as_bucket_view=True,
        )
    elif strategy_name == "fsdp":
        kwargs["strategy"] = FSDPStrategy(timeout=timedelta(hours=DDP_TIMEOUT_HOURS))
    elif strategy_name == "deepspeed":
        kwargs["strategy"] = DeepSpeedStrategy()
    elif strategy_name == "single":
        device_id: int = int(cfg_section.device_id)
        kwargs = {
            "accelerator": "cuda",
            "devices": [device_id],
            "strategy": "auto",
        }
    else:
        raise ValueError(f"Invalid GPU strategy: {strategy_name}")

    return kwargs


def resolve_precision(cfg_section: DictConfig) -> str:
    """Adjust precision based on device capabilities."""
    precision: str = str(cfg_section.precision)
    if cfg_section.use_cpu and precision == "16-mixed":
        return "bf16-mixed"
    if (
        not cfg_section.use_cpu
        and "bf16" in precision
        and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed"
    return precision
