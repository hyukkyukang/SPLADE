import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

from src.data.collator import UniversalCollator
from src.data.dataclass import TrainingDataItem
from src.data.lens_formatting import (
    build_doc_pooling_mask,
    build_query_pooling_mask,
    format_query_text,
    validate_lens_tokenizer,
)
from src.model.pl_module.train import SPLADETrainingModule
from src.utils.model_utils import build_splade_model
from src.utils.transformers import build_tokenizer


class _ListDataset(Dataset[TrainingDataItem]):
    def __init__(self, items: list[TrainingDataItem]) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> TrainingDataItem:
        return self._items[index]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU smoke test for the LENS runtime: tiny Mistral artifact build, "
            "encode pass, and one-step train/validation loop."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/smoke/lens_cpu_smoke_script",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--keep-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-build",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-train-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_command(*, repo_root: Path, command: list[str]) -> None:
    printable: str = " ".join(command)
    print(printable)
    subprocess.run(command, cwd=str(repo_root), check=True)


def _build_tiny_base_model(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab: dict[str, int] = {
        "[PAD]": 0,
        "[UNK]": 1,
        "<s>": 2,
        "</s>": 3,
        "hello": 4,
        "retrieval": 5,
        "document": 6,
        "this": 7,
        "is": 8,
        "relevant": 9,
        "query": 10,
        "for": 11,
        "testing": 12,
        "lens": 13,
        "cpu": 14,
        "smoke": 15,
        "run": 16,
        "tiny": 17,
        "model": 18,
        "positive": 19,
        "negative": 20,
        "passage": 21,
        "match": 22,
        "noise": 23,
        "train": 24,
        "validate": 25,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(output_dir)

    config = MistralConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = MistralForCausalLM(config)
    model.save_pretrained(output_dir)
    return output_dir


def _build_model_cfg(clustered_dir: Path, log_dir: Path) -> DictConfig:
    model_cfg = OmegaConf.merge(
        OmegaConf.load("config/model/_base.yaml"),
        OmegaConf.load("config/model/lens_mistral_cluster4k.yaml"),
    )
    training_cfg = OmegaConf.merge(
        OmegaConf.load("config/training/_base.yaml"),
        OmegaConf.load("config/optimizer/adam.yaml"),
        OmegaConf.load("config/training/lens_mistral.yaml"),
    )
    model_cfg.huggingface_name = str(clustered_dir)
    model_cfg.dtype = "float32"
    training_cfg.use_cpu = True
    training_cfg.precision = "32-true"
    training_cfg.strategy = "single"
    training_cfg.num_devices = 1
    training_cfg.num_workers = 0
    training_cfg.prefetch_factor = 2
    training_cfg.torch_compile = False
    training_cfg.torch_compile_loss = False
    training_cfg.batch_size = 1
    training_cfg.eval_batch_size = 1
    training_cfg.grad_accumulation = 1
    training_cfg.max_steps = 1
    training_cfg.val_check_interval = 1
    training_cfg.log_every_n_steps = 1
    training_cfg.limit_val_batches = 1.0

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "log_dir": str(log_dir),
            "root_dir_path": ".",
            "nanobeir": {
                "enabled": False,
                "run_every_n_val": 1,
                "datasets": [],
                "batch_size": 1,
                "max_seq_length": 32,
                "use_cpu": True,
                "cache_dir": None,
                "save_json": False,
            },
        }
    )
    cfg.model = model_cfg
    cfg.training = training_cfg
    return cfg


def _build_training_item(
    *,
    data_idx: int,
    tokenizer: PreTrainedTokenizerFast,
    model_cfg: DictConfig,
    query_text: str,
    positive_text: str,
    negative_text: str,
) -> TrainingDataItem:
    formatted_query: str = format_query_text(query_text, model_cfg)
    query_batch = tokenizer([formatted_query], return_tensors="pt", padding=True)
    doc_batch = tokenizer(
        [positive_text, negative_text], return_tensors="pt", padding=True
    )
    query_input_ids: torch.Tensor = query_batch["input_ids"][0]
    query_attention_mask: torch.Tensor = query_batch["attention_mask"][0]
    query_pooling_mask: torch.Tensor = build_query_pooling_mask(
        query_input_ids,
        query_attention_mask,
        tokenizer,
        model_cfg,
    )
    doc_input_ids: torch.Tensor = doc_batch["input_ids"]
    doc_attention_mask: torch.Tensor = doc_batch["attention_mask"]
    doc_pooling_mask: torch.Tensor = build_doc_pooling_mask(
        doc_attention_mask,
        model_cfg,
    )
    doc_mask: torch.Tensor = torch.tensor([True, True], dtype=torch.bool)
    pos_mask: torch.Tensor = torch.tensor([True, False], dtype=torch.bool)
    teacher_scores: torch.Tensor = torch.tensor([1.0, 0.0], dtype=torch.float32)
    labels: torch.Tensor = torch.tensor([1.0, 0.0], dtype=torch.float32)
    pos_scores: torch.Tensor = torch.tensor([1.0], dtype=torch.float32)
    neg_scores: torch.Tensor = torch.tensor([0.0], dtype=torch.float32)
    return TrainingDataItem(
        data_idx=int(data_idx),
        qid=f"q{data_idx}",
        pos_ids=[f"d{data_idx}_pos"],
        neg_ids=[f"d{data_idx}_neg"],
        query_text=query_text,
        doc_texts=[positive_text, negative_text],
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        query_pooling_mask=query_pooling_mask,
        doc_input_ids=doc_input_ids,
        doc_attention_mask=doc_attention_mask,
        doc_pooling_mask=doc_pooling_mask,
        doc_mask=doc_mask,
        pos_mask=pos_mask,
        teacher_scores=teacher_scores,
        labels=labels,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
    )


def _run_encode_smoke(cfg: DictConfig, tokenizer: PreTrainedTokenizerFast) -> dict[str, Any]:
    model = build_splade_model(cfg, use_cpu=True)
    model.eval()
    validate_lens_tokenizer(tokenizer, cfg.model)
    query_text: str = format_query_text("hello retrieval query", cfg.model)
    doc_text: str = "this document is relevant for retrieval"
    query_batch = tokenizer([query_text], return_tensors="pt", padding=True)
    doc_batch = tokenizer([doc_text], return_tensors="pt", padding=True)
    query_pooling_mask = build_query_pooling_mask(
        query_batch["input_ids"],
        query_batch["attention_mask"],
        tokenizer,
        cfg.model,
    )
    doc_pooling_mask = build_doc_pooling_mask(doc_batch["attention_mask"], cfg.model)
    with torch.no_grad():
        query_emb = model.encode_queries(
            query_batch["input_ids"],
            query_batch["attention_mask"],
            pooling_mask=query_pooling_mask,
        )
        doc_emb = model.encode_docs(
            doc_batch["input_ids"],
            doc_batch["attention_mask"],
            pooling_mask=doc_pooling_mask,
        )
        score = torch.matmul(query_emb, doc_emb.T)
    return {
        "peft_enabled": bool(model.peft_enabled),
        "compact_head_alignment": model.encoder.compact_head_alignment,
        "vocab_size": int(model.encoder.vocab_size),
        "query_embedding_shape": list(query_emb.shape),
        "doc_embedding_shape": list(doc_emb.shape),
        "query_pool_nonzero": int(query_pooling_mask.sum().item()),
        "doc_pool_nonzero": int(doc_pooling_mask.sum().item()),
        "score": float(score.item()),
        "query_tokens": tokenizer.convert_ids_to_tokens(
            query_batch["input_ids"][0].tolist()
        ),
        "query_pool_mask": query_pooling_mask[0].tolist(),
    }


def _run_one_step_train_loop(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, Any]:
    collator = UniversalCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        require_teacher_scores=False,
        max_padding=False,
    )
    train_items: list[TrainingDataItem] = [
        _build_training_item(
            data_idx=0,
            tokenizer=tokenizer,
            model_cfg=cfg.model,
            query_text="hello retrieval query",
            positive_text="this document is relevant for retrieval",
            negative_text="noise negative passage",
        )
    ]
    val_items: list[TrainingDataItem] = [
        _build_training_item(
            data_idx=1,
            tokenizer=tokenizer,
            model_cfg=cfg.model,
            query_text="cpu smoke validate query",
            positive_text="relevant document match",
            negative_text="noise negative passage",
        )
    ]
    train_loader = DataLoader(
        _ListDataset(train_items),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        _ListDataset(val_items),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    module = SPLADETrainingModule(cfg=cfg)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        default_root_dir=str(cfg.log_dir),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision="32-true",
        max_steps=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    callback_metrics: dict[str, float] = {}
    for name, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            callback_metrics[str(name)] = float(value.detach().cpu().item())
    if "val_loss" not in callback_metrics:
        raise RuntimeError(
            "One-step CPU training smoke completed without val_loss in callback metrics."
        )
    if not any(name.startswith("val_MRR_") for name in callback_metrics):
        raise RuntimeError(
            "One-step CPU training smoke completed without validation ranking metrics."
        )
    return {
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
        "callback_metrics": callback_metrics,
    }


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    output_dir = repo_root / str(args.output_dir)
    if output_dir.exists() and not bool(args.keep_output):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiny_base_dir = output_dir / "tiny_base"
    tiny_backbone_dir = output_dir / "tiny_backbone"
    tiny_clustered_dir = output_dir / "tiny_clustered"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not bool(args.skip_build):
        _build_tiny_base_model(tiny_base_dir)
        python_bin: str = sys.executable
        _run_command(
            repo_root=repo_root,
            command=[
                python_bin,
                "script/model_creation/lens/build_hf_backbone.py",
                "--base-model",
                str(tiny_base_dir),
                "--output-dir",
                str(tiny_backbone_dir),
                "--dtype",
                "float32",
                "--device",
                "cpu",
                "--local-files-only",
            ],
        )
        _run_command(
            repo_root=repo_root,
            command=[
                python_bin,
                "script/model_creation/lens/build_clustered_head.py",
                "--model-dir",
                str(tiny_backbone_dir),
                "--output-dir",
                str(tiny_clustered_dir),
                "--cluster-count",
                str(int(args.cluster_count)),
                "--backend",
                "sklearn",
                "--dtype",
                "float32",
                "--local-files-only",
            ],
        )

    cfg = _build_model_cfg(tiny_clustered_dir, log_dir)
    tokenizer = build_tokenizer(
        str(tiny_clustered_dir),
        use_fast_tokenizer=bool(cfg.model.use_fast_tokenizer),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        require_fast_tokenizer=bool(cfg.model.require_fast_tokenizer),
        local_files_only=True,
    )
    encode_summary = _run_encode_smoke(cfg, tokenizer)
    train_summary: dict[str, Any] | None = None
    if not bool(args.skip_train_loop):
        train_summary = _run_one_step_train_loop(cfg, tokenizer)

    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "cluster_count": int(args.cluster_count),
        "encode": encode_summary,
        "train_loop": train_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
