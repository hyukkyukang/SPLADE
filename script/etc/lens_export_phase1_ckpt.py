"""Convert a Phase 1 LENS Lightning checkpoint into an HF-format directory
that ``load_official_lens`` can consume.

Loads ``last.ckpt`` from a LENSTrainingModule run, rebuilds the LoRA-wrapped
backbone the same way training did, loads the trained state dict, merges the
LoRA into the base weights, and writes:

  - ``model.safetensors`` + ``config.json`` (HF format, merged backbone)
  - ``lm_head.pth``                          (clustered head as nn.Linear)
  - tokenizer + LENS cluster artifacts copied from the base artifact dir

After this, run::

    python script/evaluate_lens_mteb_parallel.py \\
        --model_name_or_path <out_dir>
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lens_export_phase1")


# Keys that are EXPECTED to be missing from the wrapped model when loading
# a LoRA-trained ckpt. After we swap in the clustered head pre-LoRA (matching
# what training did), there should be NO missing keys; this set is kept as a
# safety net for older ckpts that happened to omit ``lm_head.weight`` from the
# state dict.
_ALLOWED_MISSING_KEYS: frozenset[str] = frozenset({
    "base_model.model.lm_head.weight",
})

# Fallback LENS LoRA target modules (used only if we cannot infer from the
# ckpt's tensor names — every PEFT-trained ckpt should give us the real set).
_FALLBACK_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "down_proj", "up_proj",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to Lightning checkpoint (last.ckpt).")
    p.add_argument("--base-artifact", required=True,
                   help="Path to the LENS 4k-cluster artifact dir used as init.")
    p.add_argument("--out-dir", required=True,
                   help="Destination HF dir for merged model.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--lora-r", type=int, default=None,
                   help="Override inferred LoRA rank. Default: infer from ckpt.")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Override inferred LoRA alpha. Default: 2 * r.")
    p.add_argument("--lora-dropout", type=float, default=0.1,
                   help="LoRA dropout (not stored in ckpt — provide if differs).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the output dir even if it is non-empty.")
    return p.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def _infer_lora_rank(state_dict: dict[str, Any]) -> int:
    """Infer LoRA rank from all ``lora_A.default.weight`` tensors and verify
    they agree.

    LoRA decomposes a [out, in] linear into ``lora_A: [r, in]`` and
    ``lora_B: [out, r]``. The rank ``r`` is therefore ``lora_A.shape[0]``.
    Different layers usually share the same rank — if they don't, the ckpt
    is likely from a custom training that we shouldn't auto-handle.
    """
    ranks: set[int] = set()
    for k, v in state_dict.items():
        if "lora_A.default.weight" in k:
            ranks.add(int(v.shape[0]))
    if not ranks:
        raise ValueError(
            "No `lora_A.default.weight` tensors found in ckpt — cannot infer "
            "LoRA rank. Pass --lora-r explicitly."
        )
    if len(ranks) > 1:
        raise ValueError(
            f"LoRA tensors in ckpt have inconsistent ranks {sorted(ranks)!r}. "
            "Pass --lora-r explicitly if you know which one is correct."
        )
    return next(iter(ranks))


def _resolve_lora_hyperparams(
    state_dict: dict[str, Any],
    *,
    rank_override: int | None,
    alpha_override: int | None,
    dropout: float,
) -> tuple[int, int, float, bool]:
    """Returns ``(rank, alpha, dropout, alpha_was_inferred)``.

    ``alpha`` is NOT recoverable from a Lightning ckpt — Lightning's
    ``hyper_parameters`` only stores ``LightningModule.__init__`` args, and
    PEFT's ``adapter_config.json`` is not saved alongside. We assume the
    LENS recipe convention ``alpha = 2 * rank`` when not provided. The
    boolean return tells the caller to log a clear warning so a future user
    notices if their training deviated from convention.
    """
    rank: int = int(rank_override) if rank_override else _infer_lora_rank(state_dict)
    alpha_was_inferred: bool = alpha_override is None
    alpha: int = int(alpha_override) if alpha_override else 2 * rank
    return rank, alpha, dropout, alpha_was_inferred


def _infer_lora_target_modules(state_dict: dict[str, Any]) -> tuple[str, ...]:
    """Extract the unique target module names from the ckpt's LoRA tensors.

    LoRA paths look like ``...layers.<i>.<module>.lora_A.default.weight``;
    the ``<module>`` name is the segment immediately before ``lora_A``.
    Falls back to the known LENS recipe if we can't find any.
    """
    targets: set[str] = set()
    for k in state_dict.keys():
        if "lora_A.default.weight" not in k:
            continue
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "lora_A" and i > 0:
                targets.add(parts[i - 1])
                break
    if not targets:
        return _FALLBACK_LORA_TARGET_MODULES
    return tuple(sorted(targets))


def _check_state_dict_load(missing: list[str], unexpected: list[str]) -> None:
    real_missing = sorted(set(missing) - _ALLOWED_MISSING_KEYS)
    if real_missing:
        sample = "\n  ".join(real_missing[:20])
        raise RuntimeError(
            f"Unexpected missing keys ({len(real_missing)}) when loading ckpt "
            f"into wrapped model. First {min(20, len(real_missing))}:\n  {sample}"
        )
    if unexpected:
        sample = "\n  ".join(sorted(unexpected)[:20])
        raise RuntimeError(
            f"Unexpected keys ({len(unexpected)}) in ckpt that don't exist in "
            f"wrapped model. First {min(20, len(unexpected))}:\n  {sample}"
        )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from peft import LoraConfig, TaskType, get_peft_model

    from src.model.retriever.sparse.neural.bidirectional_mistral import (
        MistralBiForCausalLM,
    )

    dtype = _resolve_dtype(args.dtype)
    base_artifact = Path(args.base_artifact).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"Output dir {out_dir} is not empty. Pass --force to overwrite."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Lightning ckpt: %s", args.ckpt)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd_full = ck["state_dict"]
    logger.info("ckpt state_dict size = %d (epoch=%s, global_step=%s)",
                len(sd_full), ck.get("epoch"), ck.get("global_step"))

    rank, alpha, dropout, alpha_was_inferred = _resolve_lora_hyperparams(
        sd_full,
        rank_override=args.lora_r,
        alpha_override=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if alpha_was_inferred:
        logger.warning(
            "LoRA alpha=%d was INFERRED as 2*r=%d (LENS convention). "
            "Lightning ckpts do not preserve PEFT alpha. If the source "
            "training used a different alpha, the merge will scale the LoRA "
            "delta incorrectly — pass --lora-alpha explicitly to override.",
            alpha, rank,
        )
    target_modules = _infer_lora_target_modules(sd_full)

    logger.info("Loading base backbone from %s (dtype=%s)", base_artifact, dtype)
    backbone = MistralBiForCausalLM.from_pretrained(
        str(base_artifact),
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    # Match training: swap the clustered head BEFORE LoRA wrapping so the
    # state-dict shapes line up with what was saved. Without this, the export
    # builds a 32003-dim head and the ckpt's 4000-dim lm_head.weight either
    # mismatches (size error) or gets silently rejected as `unexpected_keys`.
    from src.utils.compact_head import swap_clustered_head_from_artifact
    swapped: bool = swap_clustered_head_from_artifact(backbone, str(base_artifact))
    if swapped:
        logger.info(
            "Swapped lm_head with clustered head from %s (out_features=%d, "
            "config.vocab_size=%d retained for embed_tokens compatibility)",
            base_artifact, int(backbone.lm_head.out_features),
            int(backbone.config.vocab_size),
        )
    else:
        logger.info(
            "No clustered head artifact under %s; keeping original lm_head "
            "(vocab_size=%d)",
            base_artifact, int(backbone.config.vocab_size),
        )

    logger.info(
        "Wrapping with LoRA (r=%d, alpha=%d, dropout=%.2f, targets=%s)",
        rank, alpha, dropout, list(target_modules),
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    wrapped = get_peft_model(backbone, lora_cfg)

    prefix = "encoder.model."
    sd_inner = {
        k[len(prefix):]: v
        for k, v in sd_full.items()
        if k.startswith(prefix)
    }
    logger.info("Stripped '%s' prefix; loading %d tensors into wrapped model",
                prefix, len(sd_inner))

    missing, unexpected = wrapped.load_state_dict(sd_inner, strict=False)
    _check_state_dict_load(list(missing), list(unexpected))

    logger.info("Merging LoRA into base weights")
    merged = wrapped.merge_and_unload()
    merged.eval()

    logger.info("Saving merged HF model to %s", out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    logger.info("Extracting lm_head -> %s/lm_head.pth", out_dir)
    lm_head = merged.lm_head
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"merged.lm_head is {type(lm_head)!r}; expected nn.Linear")
    head_clone = nn.Linear(
        in_features=lm_head.in_features,
        out_features=lm_head.out_features,
        bias=lm_head.bias is not None,
    )
    head_clone.weight.data.copy_(lm_head.weight.data.to(dtype=dtype))
    if lm_head.bias is not None:
        head_clone.bias.data.copy_(lm_head.bias.data.to(dtype=dtype))
    torch.save(head_clone, str(out_dir / "lm_head.pth"))

    logger.info("Copying tokenizer + LENS cluster artifacts from %s", base_artifact)
    extras = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "lens_cluster_assignments.npy",
        "lens_cluster_metadata.json",
        "lens_cluster_source_token_ids.npy",
        "splade_compact_head.pt",
        "generation_config.json",
    ]
    for name in extras:
        src = base_artifact / name
        if src.is_file():
            dst = out_dir / name
            shutil.copy2(src, dst)
            logger.info("  copied %s (%d bytes)", name, dst.stat().st_size)

    logger.info("Done. Use --model_name_or_path %s with the MTEB runner.", out_dir)


if __name__ == "__main__":
    main()
