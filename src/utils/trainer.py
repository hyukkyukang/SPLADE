from datetime import timedelta
from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from omegaconf import DictConfig

DDP_TIMEOUT_HOURS: int = 1


def get_cpu_trainer_kwargs(cfg_section: DictConfig) -> dict[str, Any]:
    """Build trainer kwargs for CPU execution."""
    strategy_name: str = str(cfg_section.strategy)
    num_devices_value: Any = cfg_section.get("num_devices")
    num_devices: int = 1 if num_devices_value is None else int(num_devices_value)
    kwargs: dict[str, Any] = {"accelerator": "cpu", "devices": num_devices}

    if strategy_name == "ddp":
        if num_devices > 1:
            use_static_graph: bool = bool(cfg_section.static_graph)
            find_unused_parameters: bool = bool(
                cfg_section.get("find_unused_parameters", False)
            )
            if find_unused_parameters:
                use_static_graph = False
            kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
                static_graph=use_static_graph,
                find_unused_parameters=find_unused_parameters,
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
    num_devices_value: Any = cfg_section.get("num_devices")
    if num_devices_value is not None:
        num_devices = min(int(num_devices_value), detected_devices)

    kwargs: dict[str, Any] = {"accelerator": "cuda", "devices": num_devices}

    if strategy_name == "ddp":
        use_static_graph: bool = bool(cfg_section.static_graph)
        find_unused_parameters: bool = bool(
            cfg_section.get("find_unused_parameters", False)
        )
        # torch.compile + DDP static graph can trigger reducer hook asserts
        # in cudagraph-heavy compile modes. Use a safer dynamic DDP setup by
        # default unless the user explicitly disables this safeguard.
        compile_enabled: bool = bool(cfg_section.get("torch_compile", False))
        compile_mode: str = str(cfg_section.get("torch_compile_mode", "default"))
        compile_mode_normalized: str = compile_mode.strip().lower()
        compile_safe_modes: set[str] = {
            "max-autotune",
            "max-autotune-no-cudagraphs",
            "reduce-overhead",
        }
        compile_ddp_safe_mode: bool = bool(
            cfg_section.get("torch_compile_ddp_safe_mode", True)
        )
        if (
            compile_enabled
            and compile_ddp_safe_mode
            and compile_mode_normalized in compile_safe_modes
        ):
            find_unused_parameters = True
            use_static_graph = False
        gradient_as_bucket_view: bool = bool(
            cfg_section.get("gradient_as_bucket_view", True)
        )
        if find_unused_parameters:
            use_static_graph = False
        kwargs["strategy"] = DDPStrategy(
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
            static_graph=use_static_graph,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
    elif strategy_name == "fsdp":
        kwargs["strategy"] = FSDPStrategy(timeout=timedelta(hours=DDP_TIMEOUT_HOURS))
    elif strategy_name == "deepspeed":
        kwargs["strategy"] = DeepSpeedStrategy()
    elif strategy_name == "single":
        kwargs = {"accelerator": "cuda", "devices": 1, "strategy": "auto"}
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
