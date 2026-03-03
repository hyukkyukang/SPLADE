"""
ANNA checkpoint loading for conversion only.

Use these helpers in script/preprocess/anna/convert_to_hf.py to load the
source ANNA model and tokenizer. At runtime, use the exported HF directory
with AutoModelForMaskedLM.from_pretrained and
AutoTokenizer.from_pretrained(..., trust_remote_code=True).
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForPreTraining,
    BertForSequenceClassification,
    BertModel,
)

_ANNA_CONFIG_FILENAME: str = "config.json"
_ANNA_TOKENIZER_MODULE_FILENAME: str = "anna_final_tokenization3.py"
_ANNA_VOCAB_FILENAME: str = "vocab.txt"
_ANNA_WEIGHT_FILENAMES: tuple[str, ...] = (
    "pytorch_model.bin",
    "model.safetensors",
)


def _resolve_model_dir(directory_path: str | Path) -> Path:
    model_dir: Path = Path(directory_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist or is not a directory: {model_dir}"
        )
    return model_dir


def _require_file(model_dir: Path, filename: str) -> Path:
    target_path: Path = model_dir / filename
    if not target_path.is_file():
        raise FileNotFoundError(f"Required file not found: {target_path}")
    return target_path


def _resolve_weights_file(model_dir: Path) -> Path:
    for filename in _ANNA_WEIGHT_FILENAMES:
        candidate: Path = model_dir / filename
        if candidate.is_file():
            return candidate
    expected: str = ", ".join(_ANNA_WEIGHT_FILENAMES)
    raise FileNotFoundError(
        f"No model weights found in {model_dir}. Expected one of: {expected}"
    )


def _load_python_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build module spec for {module_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_dict(
    weights_path: Path, map_location: str | torch.device
) -> dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading .safetensors requires the `safetensors` package."
            ) from exc
        state_dict_from_safe = load_file(str(weights_path), device="cpu")
        return dict(state_dict_from_safe)

    raw_checkpoint: Any = torch.load(str(weights_path), map_location=map_location)
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        state_candidate: Any = raw_checkpoint["state_dict"]
    else:
        state_candidate = raw_checkpoint

    if not isinstance(state_candidate, dict):
        raise ValueError(
            f"Unexpected checkpoint format in {weights_path}. "
            "Expected a state dict or a checkpoint with `state_dict`."
        )

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_candidate.items():
        if isinstance(value, torch.Tensor):
            state_dict[str(key)] = value

    if not state_dict:
        raise ValueError(
            f"No tensor parameters were found in checkpoint: {weights_path}"
        )
    return state_dict


def _build_key_transform_attempts(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    attempts: list[tuple[str, dict[str, torch.Tensor]]] = [("as-is", state_dict)]
    prefixes: tuple[str, ...] = ("model.module.", "model.", "module.")

    def strip_prefix(
        source_state_dict: dict[str, torch.Tensor], prefix: str
    ) -> tuple[dict[str, torch.Tensor], bool]:
        transformed: dict[str, torch.Tensor] = {}
        changed: bool = False
        for key, value in source_state_dict.items():
            if key.startswith(prefix):
                transformed[key[len(prefix) :]] = value
                changed = True
            else:
                transformed[key] = value
        return transformed, changed

    def normalize_attention_keys(
        source_state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        transformed: dict[str, torch.Tensor] = {}
        changed: bool = False
        for key, value in source_state_dict.items():
            normalized_key: str = key.replace(".attention_v1.", ".attention.")
            if normalized_key != key:
                changed = True
            transformed[normalized_key] = value
        return transformed, changed

    def drop_position_id_buffers(
        source_state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        transformed: dict[str, torch.Tensor] = {}
        changed: bool = False
        for key, value in source_state_dict.items():
            if key.endswith(".position_ids"):
                changed = True
                continue
            transformed[key] = value
        return transformed, changed

    for prefix in prefixes:
        prefixed_state_dict, changed = strip_prefix(state_dict, prefix)
        if changed:
            attempts.append((f"strip `{prefix}` prefix", prefixed_state_dict))

    expanded_attempts: list[tuple[str, dict[str, torch.Tensor]]] = []
    for attempt_name, candidate_state_dict in attempts:
        expanded_attempts.append((attempt_name, candidate_state_dict))
        attention_fixed_state_dict, attention_changed = normalize_attention_keys(
            candidate_state_dict
        )
        if attention_changed:
            expanded_attempts.append(
                (
                    f"{attempt_name} + rename `attention_v1`",
                    attention_fixed_state_dict,
                )
            )

    final_attempts: list[tuple[str, dict[str, torch.Tensor]]] = []
    for attempt_name, candidate_state_dict in expanded_attempts:
        final_attempts.append((attempt_name, candidate_state_dict))
        dropped_buffer_state_dict, dropped = drop_position_id_buffers(
            candidate_state_dict
        )
        if dropped:
            final_attempts.append(
                (
                    f"{attempt_name} + drop `*.position_ids`",
                    dropped_buffer_state_dict,
                )
            )
    return final_attempts


def _infer_model_load_order(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, type[torch.nn.Module]]]:
    has_pretraining_head: bool = False
    has_mlm_head: bool = False
    has_sequence_cls_head: bool = False

    for key in state_dict:
        if key.startswith("cls.seq_relationship."):
            has_pretraining_head = True
        if key.startswith("cls.predictions."):
            has_mlm_head = True
        if key.startswith("classifier."):
            has_sequence_cls_head = True

    if has_pretraining_head:
        return [
            ("BertForPreTraining", BertForPreTraining),
            ("BertForMaskedLM", BertForMaskedLM),
            ("BertModel", BertModel),
            ("BertForSequenceClassification", BertForSequenceClassification),
        ]
    if has_mlm_head:
        return [
            ("BertForMaskedLM", BertForMaskedLM),
            ("BertForPreTraining", BertForPreTraining),
            ("BertModel", BertModel),
            ("BertForSequenceClassification", BertForSequenceClassification),
        ]
    if has_sequence_cls_head:
        return [
            ("BertForSequenceClassification", BertForSequenceClassification),
            ("BertModel", BertModel),
            ("BertForMaskedLM", BertForMaskedLM),
            ("BertForPreTraining", BertForPreTraining),
        ]
    return [
        ("BertModel", BertModel),
        ("BertForSequenceClassification", BertForSequenceClassification),
        ("BertForMaskedLM", BertForMaskedLM),
        ("BertForPreTraining", BertForPreTraining),
    ]


def load_anna_model(
    directory_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Load the ANNA model from a local directory.

    Required files:
      - config.json
      - one of: pytorch_model.bin, model.safetensors,
        ANNA_ANNA_pytorch_pytorch_model.bin
    """
    model_dir: Path = _resolve_model_dir(directory_path)
    config_path: Path = _require_file(model_dir, _ANNA_CONFIG_FILENAME)
    weights_path: Path = _resolve_weights_file(model_dir)

    config: BertConfig = BertConfig.from_json_file(str(config_path))
    state_dict: dict[str, torch.Tensor] = _load_state_dict(
        weights_path, map_location=map_location
    )
    model_load_order: list[tuple[str, type[torch.nn.Module]]] = _infer_model_load_order(
        state_dict
    )

    load_errors: list[str] = []
    for attempt_name, candidate_state_dict in _build_key_transform_attempts(state_dict):
        for model_name, model_class in model_load_order:
            model: torch.nn.Module = model_class(config)
            try:
                model.load_state_dict(candidate_state_dict, strict=True)
                model.eval()
                return model
            except RuntimeError as exc:
                first_line: str = str(exc).splitlines()[0]
                load_errors.append(f"{model_name} with {attempt_name}: {first_line}")

    error_summary: str = "\n".join(load_errors)
    raise RuntimeError(
        "Failed to load ANNA model state dict with automatic format detection.\n"
        f"Tried variants:\n{error_summary}"
    )


def load_anna_masked_lm_model(
    directory_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    tie_word_embeddings: bool | None = None,
) -> BertForMaskedLM:
    """
    Load the ANNA checkpoint as a BertForMaskedLM model.

    This helper is intended for SPLADE training where MLM logits are required.
    """
    model_dir: Path = _resolve_model_dir(directory_path)
    config_path: Path = _require_file(model_dir, _ANNA_CONFIG_FILENAME)
    weights_path: Path = _resolve_weights_file(model_dir)
    config: BertConfig = BertConfig.from_json_file(str(config_path))
    if tie_word_embeddings is not None:
        config.tie_word_embeddings = bool(tie_word_embeddings)

    state_dict: dict[str, torch.Tensor] = _load_state_dict(
        weights_path, map_location=map_location
    )
    load_errors: list[str] = []
    for attempt_name, candidate_state_dict in _build_key_transform_attempts(state_dict):
        filtered_state_dict: dict[str, torch.Tensor] = {
            key: value
            for key, value in candidate_state_dict.items()
            if not key.startswith("cls.seq_relationship.")
            and not key.startswith("bert.pooler.")
            and not key.endswith(".position_ids")
        }
        model: BertForMaskedLM = BertForMaskedLM(config)
        try:
            model.load_state_dict(filtered_state_dict, strict=True)
            model.eval()
            return model
        except RuntimeError as exc:
            first_line: str = str(exc).splitlines()[0]
            load_errors.append(f"{attempt_name}: {first_line}")

    error_summary: str = "\n".join(load_errors)
    raise RuntimeError(
        "Failed to load ANNA checkpoint as BertForMaskedLM.\n"
        f"Tried variants:\n{error_summary}"
    )


def load_anna_tokenizer(
    directory_path: str | Path,
    *,
    do_lower_case: bool = True,
) -> Any:
    """
    Load the custom ANNA tokenizer from a local directory.

    Required files:
      - anna_final_tokenization3.py
      - vocab.txt
    """
    model_dir: Path = _resolve_model_dir(directory_path)
    tokenizer_module_path: Path = _require_file(
        model_dir, _ANNA_TOKENIZER_MODULE_FILENAME
    )
    vocab_path: Path = _require_file(model_dir, _ANNA_VOCAB_FILENAME)

    try:
        tokenizer_module: ModuleType = _load_python_module(
            tokenizer_module_path,
            module_name="anna_tokenizer_module",
        )
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise ModuleNotFoundError(
                "Loading ANNA tokenizer requires TensorFlow because "
                "`anna_final_tokenization3.py` imports `tensorflow as tf`."
            ) from exc
        raise

    try:
        tokenizer_cls = tokenizer_module.FullTokenizer
    except AttributeError as exc:
        raise AttributeError(
            f"`FullTokenizer` is not defined in {tokenizer_module_path}"
        ) from exc

    tokenizer = tokenizer_cls(
        vocab_file=str(vocab_path),
        do_lower_case=do_lower_case,
    )
    return tokenizer


__all__: list[str] = [
    "load_anna_model",
    "load_anna_masked_lm_model",
    "load_anna_tokenizer",
]
