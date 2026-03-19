from dataclasses import dataclass
from typing import Any, Iterable

from omegaconf import DictConfig
import torch
from torch import nn
from transformers import PreTrainedModel

from src.utils.normalize import normalize_optional_str

_DEFAULT_LORA_TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "bert": ("query", "key", "value", "dense"),
    "distilbert": ("q_lin", "k_lin", "v_lin", "out_lin"),
    "gemma": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gemma2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mistral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mpnet": ("q", "k", "v", "o"),
    "qwen2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "roberta": ("query", "key", "value", "dense"),
    "xlm_roberta": ("query", "key", "value", "dense"),
}


@dataclass(frozen=True)
class PeftSettings:
    enabled: bool
    method: str
    task_type: str
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    fan_in_fan_out: bool
    inference_mode: bool


def _cfg_get(cfg: DictConfig | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized: str | None = normalize_optional_str(value)
        if normalized is None:
            return ()
        return tuple(
            item.strip() for item in normalized.split(",") if item.strip()
        )
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def is_peft_enabled(peft_cfg: DictConfig | None) -> bool:
    return bool(_cfg_get(peft_cfg, "enabled", False))


def _is_causal_lm_loader(huggingface_model_class: str) -> bool:
    normalized: str = str(huggingface_model_class).strip()
    return normalized.endswith("ForCausalLM") or normalized == "AutoModelForCausalLM"


def resolve_peft_settings(
    peft_cfg: DictConfig | None,
    *,
    model_type: str | None,
    huggingface_model_class: str,
) -> PeftSettings:
    if not is_peft_enabled(peft_cfg):
        return PeftSettings(
            enabled=False,
            method="",
            task_type="",
            rank=0,
            alpha=0,
            dropout=0.0,
            bias="none",
            target_modules=(),
            modules_to_save=(),
            fan_in_fan_out=False,
            inference_mode=False,
        )

    method: str = str(_cfg_get(peft_cfg, "method", "lora")).strip().lower()
    if method != "lora":
        raise ValueError(f"Unsupported model.peft.method: {method!r}")

    raw_task_type: str | None = normalize_optional_str(_cfg_get(peft_cfg, "task_type"))
    if raw_task_type is None or raw_task_type.lower() == "auto":
        if _is_causal_lm_loader(huggingface_model_class):
            task_type: str = "CAUSAL_LM"
        else:
            task_type = "FEATURE_EXTRACTION"
    else:
        task_type = raw_task_type.upper()

    resolved_model_type: str | None = normalize_optional_str(model_type)
    default_target_modules: tuple[str, ...] = ()
    if resolved_model_type is not None:
        default_target_modules = _DEFAULT_LORA_TARGET_MODULES.get(
            resolved_model_type.lower(), ()
        )
    target_modules: tuple[str, ...] = _normalize_string_list(
        _cfg_get(peft_cfg, "target_modules")
    )
    if not target_modules:
        target_modules = default_target_modules
    if not target_modules:
        raise ValueError(
            "model.peft.target_modules must be set when no default mapping exists "
            f"for model_type={resolved_model_type!r}."
        )

    rank: int = int(_cfg_get(peft_cfg, "r", 16))
    alpha: int = int(_cfg_get(peft_cfg, "alpha", 32))
    dropout: float = float(_cfg_get(peft_cfg, "dropout", 0.0))
    if rank <= 0:
        raise ValueError("model.peft.r must be > 0 when PEFT is enabled.")
    if alpha <= 0:
        raise ValueError("model.peft.alpha must be > 0 when PEFT is enabled.")
    if dropout < 0.0:
        raise ValueError("model.peft.dropout must be >= 0.")

    return PeftSettings(
        enabled=True,
        method=method,
        task_type=task_type,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=str(_cfg_get(peft_cfg, "bias", "none")).strip().lower(),
        target_modules=target_modules,
        modules_to_save=_normalize_string_list(_cfg_get(peft_cfg, "modules_to_save")),
        fan_in_fan_out=bool(_cfg_get(peft_cfg, "fan_in_fan_out", False)),
        inference_mode=bool(_cfg_get(peft_cfg, "inference_mode", False)),
    )


def apply_peft_adapter(
    model: PreTrainedModel,
    *,
    settings: PeftSettings,
) -> tuple[PreTrainedModel, frozenset[str]]:
    if not settings.enabled:
        return model, frozenset()
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is enabled in the config but the `peft` package is not installed. "
            "Install it or disable model.peft.enabled."
        ) from exc

    task_type_value: Any = getattr(TaskType, settings.task_type, None)
    if task_type_value is None:
        raise ValueError(
            f"Unsupported model.peft.task_type for PEFT TaskType: {settings.task_type!r}"
        )

    lora_config = LoraConfig(
        task_type=task_type_value,
        inference_mode=settings.inference_mode,
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=list(settings.target_modules),
        modules_to_save=(
            list(settings.modules_to_save) if settings.modules_to_save else None
        ),
        fan_in_fan_out=settings.fan_in_fan_out,
    )
    wrapped_model: PreTrainedModel = get_peft_model(model, lora_config)
    trainable_parameter_names: frozenset[str] = frozenset(
        name for name, parameter in wrapped_model.named_parameters() if parameter.requires_grad
    )
    if not trainable_parameter_names:
        raise ValueError(
            "Applying PEFT produced no trainable parameters. Check target_modules."
        )
    return wrapped_model, trainable_parameter_names


def is_peft_model(model: Any) -> bool:
    return getattr(model, "peft_config", None) is not None


def unwrap_peft_model(model: nn.Module) -> nn.Module:
    current: nn.Module = model
    visited: set[int] = {id(current)}
    while True:
        next_module: nn.Module | None = None
        if is_peft_model(current):
            candidate: Any = getattr(current, "base_model", None)
            if isinstance(candidate, nn.Module):
                next_module = candidate
        elif current.__class__.__module__.startswith("peft."):
            candidate = getattr(current, "model", None)
            if isinstance(candidate, nn.Module):
                next_module = candidate
        if next_module is None or id(next_module) in visited:
            return current
        visited.add(id(next_module))
        current = next_module
