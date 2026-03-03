import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.prototype.embeddinggemma_lsr.data import (
    TextPair,
    build_text_pairs,
    collect_required_ids,
    column_names_of,
    load_hf_splits,
    lookup_texts_by_ids,
    maybe_concat_datasets,
    resolve_first_present_column,
)
from src.prototype.embeddinggemma_lsr.losses import (
    compute_ranking_metrics,
    flops_regularization,
    info_nce_in_batch,
)
from src.prototype.embeddinggemma_lsr.model import EmbeddingGemmaLSRModel


@dataclass(frozen=True)
class TrainingBatch:
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor


class _PairDataset(Dataset):
    def __init__(self, pairs: list[TextPair]) -> None:
        self._pairs: list[TextPair] = list(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> TextPair:
        return self._pairs[index]


@dataclass(frozen=True)
class _TokenizedBatchCollator:
    tokenizer: PreTrainedTokenizerBase
    max_query_length: int
    max_doc_length: int

    def __call__(self, pairs: list[TextPair]) -> TrainingBatch:
        queries: list[str] = [pair.query for pair in pairs]
        docs: list[str] = [pair.positive for pair in pairs]

        q_tok: dict[str, torch.Tensor] = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=int(self.max_query_length),
            return_tensors="pt",
        )
        d_tok: dict[str, torch.Tensor] = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=int(self.max_doc_length),
            return_tensors="pt",
        )
        return TrainingBatch(
            query_input_ids=q_tok["input_ids"],
            query_attention_mask=q_tok["attention_mask"],
            doc_input_ids=d_tok["input_ids"],
            doc_attention_mask=d_tok["attention_mask"],
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EmbeddingGemma-LSR with InfoNCE + FLOPs regularization."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--meta-hf-name", type=str, default=None)
    parser.add_argument("--meta-hf-subset", type=str, default="triplets")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--allow-missing-val-split", action="store_true")
    parser.add_argument("--hf-cache-dir", type=str, default=None)

    parser.add_argument("--meta-query-id-column", type=str, default="query_id")
    parser.add_argument("--meta-positive-id-column", type=str, default="positive_id")
    parser.add_argument("--meta-query-text-column", type=str, default="query")
    parser.add_argument("--meta-positive-text-column", type=str, default="positive")

    parser.add_argument("--query-subset", type=str, default="queries")
    parser.add_argument("--query-split", type=str, default="train")
    parser.add_argument("--query-id-column", type=str, default="query_id")
    parser.add_argument("--query-text-column", type=str, default="query")

    parser.add_argument("--corpus-subset", type=str, default="corpus")
    parser.add_argument("--corpus-split", type=str, default="train")
    parser.add_argument("--corpus-id-column", type=str, default="passage_id")
    parser.add_argument("--corpus-text-column", type=str, default="passage")

    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=5000)

    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-flops-q", type=float, default=2e-4)
    parser.add_argument("--lambda-flops-d", type=float, default=1e-4)

    parser.add_argument("--max-query-length", type=int, default=128)
    parser.add_argument("--max-doc-length", type=int, default=256)

    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-queries", type=int, default=2000)
    parser.add_argument("--eval-max-docs", type=int, default=20000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _default_values() -> dict[str, Any]:
    parser: argparse.ArgumentParser = _build_parser()
    defaults: dict[str, Any] = {}
    action: argparse.Action
    for action in parser._actions:
        if action.dest in {None, "help"}:
            continue
        defaults[str(action.dest)] = action.default
    return defaults


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    cfg = OmegaConf.load(args.config)
    payload: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    defaults: dict[str, Any] = _default_values()
    for key, value in payload.items():
        if not hasattr(args, key):
            continue
        if key in defaults and getattr(args, key) == defaults[key]:
            setattr(args, key, value)
    return args


def _resolve_device(device_value: str) -> torch.device:
    text: str = str(device_value).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    key: str = str(dtype_name).lower()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_required_args(args: argparse.Namespace) -> None:
    required_keys: tuple[str, ...] = ("model_dir", "output_dir", "meta_hf_name")
    key: str
    for key in required_keys:
        value: Any | None = getattr(args, key, None)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Missing required argument `{key}`. "
                "Provide it directly or via --config."
            )


def _resolve_effective_max_train_pairs(args: argparse.Namespace) -> int:
    configured_max: Any | None = getattr(args, "max_train_pairs", None)
    if configured_max is not None:
        value: int = int(configured_max)
        if value <= 0:
            raise ValueError("max_train_pairs must be a positive integer when provided.")
        return value

    # Prevent unbounded materialization on massive datasets like MSMARCO triplets.
    max_steps: int = max(int(args.max_steps), 1)
    batch_size: int = max(int(args.batch_size), 1)
    auto_pairs: int = max_steps * batch_size
    auto_cap: int = 500_000
    effective_pairs: int = min(auto_pairs, auto_cap)
    print(
        json.dumps(
            {
                "event": "auto_max_train_pairs",
                "configured": None,
                "computed_max_steps_x_batch_size": auto_pairs,
                "cap": auto_cap,
                "effective_max_train_pairs": effective_pairs,
            },
            ensure_ascii=False,
        )
    )
    return effective_pairs


def _unwrap_model(model: nn.Module) -> EmbeddingGemmaLSRModel:
    if isinstance(model, nn.DataParallel):
        wrapped: nn.Module = model.module
        if not isinstance(wrapped, EmbeddingGemmaLSRModel):
            raise TypeError("Unexpected wrapped model type for DataParallel.")
        return wrapped
    if not isinstance(model, EmbeddingGemmaLSRModel):
        raise TypeError("Unexpected model type.")
    return model


def _encode_with_model(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if isinstance(model, nn.DataParallel):
        return model(input_ids=input_ids, attention_mask=attention_mask)
    return _unwrap_model(model).encode(input_ids=input_ids, attention_mask=attention_mask)


def _build_linear_warmup_decay_lr_lambda(
    warmup_steps: int,
    total_steps: int,
):
    warmup: int = max(int(warmup_steps), 0)
    total: int = max(int(total_steps), 1)

    def _lr_lambda(current_step: int) -> float:
        step: int = int(current_step)
        if warmup > 0 and step < warmup:
            return float(step) / float(max(warmup, 1))
        remaining: int = max(total - step, 0)
        decay_span: int = max(total - warmup, 1)
        return float(remaining) / float(decay_span)

    return _lr_lambda


def _load_pairs_for_splits(
    *,
    args: argparse.Namespace,
    splits: list[str],
    allow_missing_split: bool,
    max_pairs: int | None,
) -> list[TextPair]:
    meta_datasets = load_hf_splits(
        hf_name=args.meta_hf_name,
        hf_subset=args.meta_hf_subset,
        splits=splits,
        cache_dir=args.hf_cache_dir,
        data_files=None,
        allow_missing_split=allow_missing_split,
    )
    if not meta_datasets:
        return []

    meta_dataset = maybe_concat_datasets(meta_datasets)
    columns: list[str] = column_names_of(meta_dataset)
    query_text_col: str | None = resolve_first_present_column(
        columns,
        [args.meta_query_text_column, "query", "question", "query_text"],
    )
    positive_text_col: str | None = resolve_first_present_column(
        columns,
        [args.meta_positive_text_column, "positive", "passage", "doc", "positive_text"],
    )

    if query_text_col is not None and positive_text_col is not None:
        return build_text_pairs(
            meta_dataset=meta_dataset,
            query_text_column=query_text_col,
            positive_text_column=positive_text_col,
            query_id_column=args.meta_query_id_column,
            positive_id_column=args.meta_positive_id_column,
            query_lookup=None,
            corpus_lookup=None,
            max_pairs=max_pairs,
        )

    query_ids, positive_ids, _ = collect_required_ids(
        meta_dataset=meta_dataset,
        query_id_column=args.meta_query_id_column,
        positive_id_column=args.meta_positive_id_column,
        max_rows=max_pairs,
    )

    query_dataset = maybe_concat_datasets(
        load_hf_splits(
            hf_name=args.meta_hf_name,
            hf_subset=args.query_subset,
            splits=[args.query_split],
            cache_dir=args.hf_cache_dir,
            data_files=None,
            allow_missing_split=False,
        )
    )
    corpus_dataset = maybe_concat_datasets(
        load_hf_splits(
            hf_name=args.meta_hf_name,
            hf_subset=args.corpus_subset,
            splits=[args.corpus_split],
            cache_dir=args.hf_cache_dir,
            data_files=None,
            allow_missing_split=False,
        )
    )

    query_lookup: dict[str, str] = lookup_texts_by_ids(
        dataset=query_dataset,
        id_column=args.query_id_column,
        text_column=args.query_text_column,
        wanted_ids=query_ids,
    )
    corpus_lookup: dict[str, str] = lookup_texts_by_ids(
        dataset=corpus_dataset,
        id_column=args.corpus_id_column,
        text_column=args.corpus_text_column,
        wanted_ids=positive_ids,
    )

    return build_text_pairs(
        meta_dataset=meta_dataset,
        query_text_column=None,
        positive_text_column=None,
        query_id_column=args.meta_query_id_column,
        positive_id_column=args.meta_positive_id_column,
        query_lookup=query_lookup,
        corpus_lookup=corpus_lookup,
        max_pairs=max_pairs,
    )


def _encode_texts(
    *,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not texts:
        base_model: EmbeddingGemmaLSRModel = _unwrap_model(model)
        return torch.empty((0, base_model.projection.out_features), device=device)

    vectors: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            chunk: list[str] = texts[start : start + int(batch_size)]
            tokens: dict[str, torch.Tensor] = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            reps: torch.Tensor = _encode_with_model(
                model,
                input_ids=tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device),
            )
            vectors.append(reps)
    return torch.cat(vectors, dim=0)


def _evaluate(
    *,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    val_pairs: list[TextPair],
    device: torch.device,
    max_query_length: int,
    max_doc_length: int,
    eval_batch_size: int,
    eval_max_queries: int,
    eval_max_docs: int,
) -> dict[str, float]:
    if not val_pairs:
        return {"mrr@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}

    sampled_pairs: list[TextPair] = val_pairs[: int(eval_max_queries)]

    doc_key_to_text: dict[str, str] = {}
    for pair in sampled_pairs:
        key: str = pair.positive_id if pair.positive_id is not None else pair.positive
        if key not in doc_key_to_text:
            doc_key_to_text[key] = pair.positive
        if len(doc_key_to_text) >= int(eval_max_docs):
            break

    filtered_pairs: list[TextPair] = []
    for pair in sampled_pairs:
        key = pair.positive_id if pair.positive_id is not None else pair.positive
        if key in doc_key_to_text:
            filtered_pairs.append(pair)

    doc_keys: list[str] = list(doc_key_to_text.keys())
    doc_texts: list[str] = [doc_key_to_text[key] for key in doc_keys]
    doc_index_by_key: dict[str, int] = {key: idx for idx, key in enumerate(doc_keys)}

    query_texts: list[str] = [pair.query for pair in filtered_pairs]
    positive_indices: list[int] = []
    for pair in filtered_pairs:
        key = pair.positive_id if pair.positive_id is not None else pair.positive
        positive_indices.append(doc_index_by_key[key])

    query_reps: torch.Tensor = _encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=query_texts,
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
    )
    doc_reps: torch.Tensor = _encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=doc_texts,
        max_length=max_doc_length,
        batch_size=eval_batch_size,
        device=device,
    )

    scores: torch.Tensor = torch.matmul(query_reps.float(), doc_reps.float().transpose(0, 1))
    metrics: dict[str, float] = compute_ranking_metrics(
        scores,
        positive_indices=torch.tensor(positive_indices, device=scores.device),
        k_values=(10,),
    )
    return metrics


def _save_checkpoint(
    *,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    name: str,
    metadata: dict[str, Any],
) -> None:
    target_dir: Path = output_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    base_model: EmbeddingGemmaLSRModel = _unwrap_model(model)
    base_model.save_pretrained(target_dir, tokenizer=tokenizer, extra_metadata=metadata)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = _apply_config_overrides(args)
    _validate_required_args(args)

    _set_seed(int(args.seed))

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device: torch.device = _resolve_device(args.device)
    dtype: torch.dtype = _resolve_dtype(args.dtype)

    model: nn.Module = EmbeddingGemmaLSRModel.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        map_location="cpu",
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model.to(device)
    if device.type == "cuda":
        available_gpu_count: int = int(torch.cuda.device_count())
        requested_gpu_count: int = max(int(args.num_gpus), 1)
        use_gpu_count: int = min(requested_gpu_count, available_gpu_count)
        if use_gpu_count > 1:
            model = nn.DataParallel(model, device_ids=list(range(use_gpu_count)))
            print(
                json.dumps(
                    {
                        "event": "multi_gpu_enabled",
                        "num_gpus": use_gpu_count,
                    },
                    ensure_ascii=False,
                )
            )
    model.train()

    effective_max_train_pairs: int = _resolve_effective_max_train_pairs(args)
    print(
        json.dumps(
            {
                "event": "loading_pairs",
                "split": args.train_split,
                "max_pairs": effective_max_train_pairs,
            },
            ensure_ascii=False,
        )
    )
    train_pairs: list[TextPair] = _load_pairs_for_splits(
        args=args,
        splits=[args.train_split],
        allow_missing_split=False,
        max_pairs=effective_max_train_pairs,
    )
    if not train_pairs:
        raise RuntimeError("No training pairs were loaded.")
    print(
        json.dumps(
            {
                "event": "train_pairs_loaded",
                "count": len(train_pairs),
            },
            ensure_ascii=False,
        )
    )

    print(
        json.dumps(
            {
                "event": "loading_pairs",
                "split": args.val_split,
                "max_pairs": args.max_val_pairs,
            },
            ensure_ascii=False,
        )
    )
    val_pairs: list[TextPair] = _load_pairs_for_splits(
        args=args,
        splits=[args.val_split],
        allow_missing_split=bool(args.allow_missing_val_split),
        max_pairs=args.max_val_pairs,
    )
    print(
        json.dumps(
            {
                "event": "val_pairs_loaded",
                "count": len(val_pairs),
            },
            ensure_ascii=False,
        )
    )

    effective_num_workers: int = int(args.num_workers)
    if effective_num_workers > 0 and sys.version_info >= (3, 14):
        print(
            json.dumps(
                {
                    "event": "num_workers_fallback",
                    "requested_num_workers": int(args.num_workers),
                    "effective_num_workers": 0,
                    "reason": "python3.14_forkserver_pickling_safety",
                },
                ensure_ascii=False,
            )
        )
        effective_num_workers = 0

    train_dataset: _PairDataset = _PairDataset(train_pairs)
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=effective_num_workers,
        collate_fn=_TokenizedBatchCollator(
            tokenizer=tokenizer,
            max_query_length=int(args.max_query_length),
            max_doc_length=int(args.max_doc_length),
        ),
        drop_last=(len(train_dataset) >= int(args.batch_size)),
        pin_memory=(device.type == "cuda"),
    )

    optimizer_name: str = str(args.optimizer).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_linear_warmup_decay_lr_lambda(
            warmup_steps=int(args.warmup_steps),
            total_steps=int(args.max_steps),
        ),
    )

    use_fp16_scaler: bool = device.type == "cuda" and dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        grad_scaler = torch.amp.GradScaler(device.type, enabled=use_fp16_scaler)
    else:
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)

    logs: list[dict[str, Any]] = []
    best_mrr10: float = float("-inf")
    global_step: int = 0

    train_iter = iter(train_loader)
    while global_step < int(args.max_steps):
        try:
            batch: TrainingBatch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)

        autocast_enabled: bool = device.type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=dtype):
            query_input_ids: torch.Tensor = batch.query_input_ids.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            query_attention_mask: torch.Tensor = batch.query_attention_mask.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            doc_input_ids: torch.Tensor = batch.doc_input_ids.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            doc_attention_mask: torch.Tensor = batch.doc_attention_mask.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            q_reps: torch.Tensor = _encode_with_model(
                model,
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            d_reps: torch.Tensor = _encode_with_model(
                model,
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
            )
            info_loss, _ = info_nce_in_batch(
                q_reps,
                d_reps,
                temperature=float(args.temperature),
            )
            q_flops: torch.Tensor = flops_regularization(q_reps)
            d_flops: torch.Tensor = flops_regularization(d_reps)
            loss: torch.Tensor = (
                info_loss
                + float(args.lambda_flops_q) * q_flops
                + float(args.lambda_flops_d) * d_flops
            )

        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % 20 == 0 or global_step == 1:
            lr_value: float = float(scheduler.get_last_lr()[0])
            log_entry: dict[str, Any] = {
                "step": global_step,
                "train/loss": float(loss.detach().cpu().item()),
                "train/info_nce": float(info_loss.detach().cpu().item()),
                "train/q_flops": float(q_flops.detach().cpu().item()),
                "train/d_flops": float(d_flops.detach().cpu().item()),
                "train/lr": lr_value,
            }
            logs.append(log_entry)
            print(json.dumps(log_entry, ensure_ascii=False))

        if (
            int(args.eval_every) > 0
            and global_step % int(args.eval_every) == 0
            and val_pairs
        ):
            metrics: dict[str, float] = _evaluate(
                model=model,
                tokenizer=tokenizer,
                val_pairs=val_pairs,
                device=device,
                max_query_length=int(args.max_query_length),
                max_doc_length=int(args.max_doc_length),
                eval_batch_size=int(args.eval_batch_size),
                eval_max_queries=int(args.eval_max_queries),
                eval_max_docs=int(args.eval_max_docs),
            )
            eval_entry: dict[str, Any] = {
                "step": global_step,
                **{f"val/{k}": v for k, v in metrics.items()},
            }
            logs.append(eval_entry)
            print(json.dumps(eval_entry, ensure_ascii=False))

            mrr10: float = float(metrics.get("mrr@10", 0.0))
            if mrr10 > best_mrr10:
                best_mrr10 = mrr10
                _save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    name="best",
                    metadata={
                        "step": global_step,
                        "best_mrr@10": best_mrr10,
                    },
                )

    final_metrics: dict[str, float] = {}
    if val_pairs:
        final_metrics = _evaluate(
            model=model,
            tokenizer=tokenizer,
            val_pairs=val_pairs,
            device=device,
            max_query_length=int(args.max_query_length),
            max_doc_length=int(args.max_doc_length),
            eval_batch_size=int(args.eval_batch_size),
            eval_max_queries=int(args.eval_max_queries),
            eval_max_docs=int(args.eval_max_docs),
        )

    _save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        name="last",
        metadata={
            "step": global_step,
            "final_metrics": final_metrics,
        },
    )

    (output_dir / "train_logs.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in logs) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "steps": global_step,
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
                "best_mrr@10": best_mrr10 if best_mrr10 != float("-inf") else None,
                "final_metrics": final_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training complete. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
