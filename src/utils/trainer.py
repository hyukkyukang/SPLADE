from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from omegaconf import DictConfig

DDP_TIMEOUT_HOURS: int = 1


@dataclass(frozen=True)
class EffectiveDistributedSettings:
    """Resolved runtime settings shared by trainer and compile policy."""

    strategy_name: str
    num_devices: int
    distributed_enabled: bool
    ddp_enabled: bool
    static_graph: bool
    find_unused_parameters: bool
    gradient_as_bucket_view: bool
    full_model_compile_safe: bool
    start_method: str | None = None


def _resolve_configured_num_devices(
    cfg_section: DictConfig, *, use_cpu: bool
) -> int:
    raw_num_devices: Any = cfg_section.get("num_devices")
    if use_cpu:
        return 1 if raw_num_devices is None else int(raw_num_devices)

    detected_devices: int = int(torch.cuda.device_count())
    if raw_num_devices is None:
        return detected_devices
    return min(int(raw_num_devices), detected_devices)


def _compile_safe_mode_requests_dynamic_ddp(cfg_section: DictConfig) -> bool:
    compile_enabled: bool = bool(cfg_section.get("torch_compile", False))
    compile_mode: str = str(cfg_section.get("torch_compile_mode", "default"))
    compile_mode_normalized: str = compile_mode.strip().lower()
    compile_safe_modes: set[str] = {
        "max-autotune",
        "reduce-overhead",
    }
    compile_ddp_safe_mode: bool = bool(
        cfg_section.get("torch_compile_ddp_safe_mode", True)
    )
    return (
        compile_enabled
        and compile_ddp_safe_mode
        and compile_mode_normalized in compile_safe_modes
    )


def resolve_effective_distributed_settings(
    cfg_section: DictConfig,
) -> EffectiveDistributedSettings:
    """Resolve runtime distributed settings after compile safety overrides."""
    strategy_name: str = str(cfg_section.strategy).lower()
    use_cpu: bool = bool(cfg_section.get("use_cpu", False))
    num_devices: int = _resolve_configured_num_devices(cfg_section, use_cpu=use_cpu)
    raw_static_graph: bool = bool(cfg_section.get("static_graph", False))
    raw_find_unused_parameters: bool = bool(
        cfg_section.get("find_unused_parameters", False)
    )
    static_graph: bool = raw_static_graph
    find_unused_parameters: bool = raw_find_unused_parameters
    gradient_as_bucket_view: bool = bool(
        cfg_section.get("gradient_as_bucket_view", True)
    )
    ddp_enabled: bool = False
    distributed_enabled: bool = False
    start_method: str | None = None

    if strategy_name == "ddp":
        distributed_enabled = num_devices > 1
        ddp_enabled = distributed_enabled
        if (
            distributed_enabled
            and not use_cpu
            and _compile_safe_mode_requests_dynamic_ddp(cfg_section)
        ):
            find_unused_parameters = True
            static_graph = False
        if find_unused_parameters:
            static_graph = False
    elif strategy_name == "ddp_spawn":
        distributed_enabled = num_devices > 1
        ddp_enabled = distributed_enabled
        start_method = "spawn" if distributed_enabled else None
        if (
            distributed_enabled
            and not use_cpu
            and _compile_safe_mode_requests_dynamic_ddp(cfg_section)
        ):
            find_unused_parameters = True
            static_graph = False
        if find_unused_parameters:
            static_graph = False
    elif strategy_name == "fsdp":
        distributed_enabled = num_devices > 1
    elif strategy_name == "deepspeed":
        distributed_enabled = num_devices > 1
    elif strategy_name == "single":
        num_devices = 1
        gradient_as_bucket_view = False
    else:
        raise ValueError(
            f"Invalid {'CPU' if use_cpu else 'GPU'} strategy: {strategy_name}"
        )

    full_model_compile_safe: bool
    if ddp_enabled:
        full_model_compile_safe = static_graph and not find_unused_parameters
    else:
        full_model_compile_safe = raw_static_graph and not raw_find_unused_parameters

    return EffectiveDistributedSettings(
        strategy_name=strategy_name,
        num_devices=num_devices,
        distributed_enabled=distributed_enabled,
        ddp_enabled=ddp_enabled,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        full_model_compile_safe=full_model_compile_safe,
        start_method=start_method,
    )


def get_cpu_trainer_kwargs(cfg_section: DictConfig) -> dict[str, Any]:
    """Build trainer kwargs for CPU execution."""
    settings: EffectiveDistributedSettings = resolve_effective_distributed_settings(
        cfg_section
    )
    strategy_name: str = settings.strategy_name
    kwargs: dict[str, Any] = {
        "accelerator": "cpu",
        "devices": settings.num_devices,
    }

    if strategy_name == "ddp":
        if settings.ddp_enabled:
            kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
                static_graph=settings.static_graph,
                find_unused_parameters=settings.find_unused_parameters,
            )
        else:
            kwargs["strategy"] = "auto"
    elif strategy_name == "ddp_spawn":
        if settings.ddp_enabled:
            kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
                static_graph=settings.static_graph,
                find_unused_parameters=settings.find_unused_parameters,
                start_method=settings.start_method,
            )
        else:
            kwargs["strategy"] = "auto"
    elif strategy_name == "single":
        kwargs["strategy"] = "auto"
    else:
        raise ValueError(f"Invalid CPU strategy: {strategy_name}")

    return kwargs


def get_gpu_trainer_kwargs(cfg_section: DictConfig) -> dict[str, Any]:
    """Build trainer kwargs for CUDA execution."""
    settings: EffectiveDistributedSettings = resolve_effective_distributed_settings(
        cfg_section
    )
    strategy_name: str = settings.strategy_name
    kwargs: dict[str, Any] = {"accelerator": "cuda", "devices": settings.num_devices}

    if strategy_name == "ddp":
        kwargs["strategy"] = DDPStrategy(
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
            static_graph=settings.static_graph,
            find_unused_parameters=settings.find_unused_parameters,
            gradient_as_bucket_view=settings.gradient_as_bucket_view,
        )
    elif strategy_name == "ddp_spawn":
        kwargs["strategy"] = DDPStrategy(
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
            static_graph=settings.static_graph,
            find_unused_parameters=settings.find_unused_parameters,
            gradient_as_bucket_view=settings.gradient_as_bucket_view,
            start_method=settings.start_method,
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
