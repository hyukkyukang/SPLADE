import argparse
import json
import os
import statistics
import tempfile
import time
from glob import glob
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from hydra import compose, initialize_config_dir

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import EncodeDataModule
from src.index.sparse import SparseShardWriter
from src.model.pl_module import SPLADEEncodeModule
from src.utils.script_setup import configure_default_entrypoint_environment


configure_default_entrypoint_environment(load_env=True, set_matmul_precision=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile DDP encode behavior across multiple GPUs."
    )
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--data-glob",
        type=str,
        default=".cache/hf/patent-us-corpus-small/data/*.parquet",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=8,
        help="Use the first N parquet files from data-glob.",
    )
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", type=str, default="default")
    parser.add_argument("--max-windows-per-forward", type=int, default=160)
    parser.add_argument(
        "--broadcast-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DDP broadcast_buffers flag.",
    )
    return parser.parse_args()


def init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def resolve_files(data_glob: str, limit_files: int | None) -> list[str]:
    files = sorted(glob(str(data_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched data glob: {data_glob}")
    if limit_files is not None and limit_files > 0:
        files = files[: int(limit_files)]
    return files


def build_cfg(args: argparse.Namespace, files: list[str], world_size: int):
    overrides = [
        "model=splade_v3_naver",
        "model.local_files_only=true",
        "dataset=patent_us_corpus_small",
        "encoding.long_doc_strategy=truncate",
        f"encoding.batch_size={int(args.batch_size)}",
        f"encoding.num_workers={int(args.num_workers)}",
        f"encoding.prefetch_factor={int(args.prefetch_factor)}",
        f"encoding.num_devices={int(world_size)}",
        "encoding.strategy=ddp",
        f"encoding.torch_compile={'true' if bool(args.torch_compile) else 'false'}",
        f"encoding.torch_compile_mode={str(args.torch_compile_mode)}",
        f"encoding.max_windows_per_forward={int(args.max_windows_per_forward)}",
        "encoding.async_write=false",
        "tag=profile_encode_ddp",
    ]
    with initialize_config_dir(version_base=None, config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="encode", overrides=overrides)
    cfg.dataset.query_corpus_hf_data_files.train = files
    os.makedirs(cfg.log_dir, exist_ok=True)
    return cfg


def setup_module(cfg) -> SPLADEEncodeModule:
    module = SPLADEEncodeModule(cfg)
    module = module.to("cuda")
    module.eval()
    raw_exclude_token_ids = module._resolve_exclude_token_ids()
    exclude_output_ids = module._resolve_exclude_output_ids(raw_exclude_token_ids)
    if exclude_output_ids:
        module._exclude_output_ids_tensor = torch.tensor(
            exclude_output_ids, dtype=torch.long, device=module.device
        )
    else:
        module._exclude_output_ids_tensor = None
    module._min_weight = float(cfg.encoding.sparse_min_weight)
    module._top_k = cfg.encoding.sparse_top_k
    return module


def wrap_doc_encoder_ddp(
    module: SPLADEEncodeModule,
    *,
    local_rank: int,
    broadcast_buffers: bool,
) -> None:
    doc_encoder = module.model._doc_encoder_fn
    if not isinstance(doc_encoder, torch.nn.Module):
        raise TypeError(
            "Document encoder function must be an nn.Module to profile DDP behavior."
        )
    ddp_doc_encoder = DistributedDataParallel(
        doc_encoder,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=bool(broadcast_buffers),
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    module.model._doc_encoder_fn = ddp_doc_encoder


def build_writer(module: SPLADEEncodeModule, cfg, output_dir: Path, rank: int) -> SparseShardWriter:
    exclude_output_ids = (
        module._exclude_output_ids_tensor.tolist()
        if module._exclude_output_ids_tensor is not None
        else []
    )
    return SparseShardWriter(
        output_dir=output_dir,
        vocab_size=int(module.model.encoder.vocab_size),
        rank=rank,
        top_k=cfg.encoding.sparse_top_k,
        min_weight=float(cfg.encoding.sparse_min_weight),
        exclude_output_ids=exclude_output_ids,
        source_exclude_token_ids=module._resolve_exclude_token_ids(),
        model_family=str(module.model.family),
        output_space=module.model.encoder.output_space,
        shard_max_docs=int(cfg.encoding.shard_max_docs),
        value_dtype=str(cfg.encoding.value_dtype),
    )


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = init_dist()
    files = resolve_files(args.data_glob, args.limit_files)
    cfg = build_cfg(args, files, world_size)

    data_module = EncodeDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.predict_dataloader()
    module = setup_module(cfg)
    wrap_doc_encoder_ddp(
        module,
        local_rank=local_rank,
        broadcast_buffers=bool(args.broadcast_buffers),
    )

    writer: SparseShardWriter | None = None
    if not bool(args.skip_write):
        output_dir = Path(tempfile.mkdtemp(prefix=f"splade-ddp-prof-rank{rank}-"))
        writer = build_writer(module, cfg, output_dir, rank)

    iterator = iter(dataloader)
    rows: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    with torch.inference_mode():
        for step in range(int(args.steps)):
            t_wait0 = time.perf_counter()
            batch = next(iterator)
            t_wait1 = time.perf_counter()

            torch.cuda.synchronize()
            t_h2d0 = time.perf_counter()
            doc_ids = list(batch["doc_ids"])
            doc_input_ids = batch["doc_input_ids"].to(module.device, non_blocking=True)
            doc_attention_mask = batch["doc_attention_mask"].to(
                module.device, non_blocking=True
            )
            doc_pooling_mask = batch.get("doc_pooling_mask")
            if doc_pooling_mask is not None:
                doc_pooling_mask = doc_pooling_mask.to(
                    module.device, non_blocking=True
                )
            doc_indptr = batch["doc_indptr"]
            torch.cuda.synchronize()
            t_h2d1 = time.perf_counter()

            t_enc0 = time.perf_counter()
            doc_reps = module._encode_and_aggregate_window_batch(
                doc_input_ids,
                doc_attention_mask,
                doc_pooling_mask,
                doc_indptr,
            )
            torch.cuda.synchronize()
            t_enc1 = time.perf_counter()

            t_sp0 = time.perf_counter()
            indptr, indices, values = module._sparsify_batch(doc_reps)
            torch.cuda.synchronize()
            t_sp1 = time.perf_counter()

            t_wr0 = time.perf_counter()
            if writer is not None:
                writer.write_sparse_csr_batch(doc_ids, indptr, indices, values)
            t_wr1 = time.perf_counter()

            if step < int(args.warmup):
                continue

            wait_s = t_wait1 - t_wait0
            h2d_s = t_h2d1 - t_h2d0
            encode_s = t_enc1 - t_enc0
            sparsify_s = t_sp1 - t_sp0
            write_s = t_wr1 - t_wr0
            total_step_s = wait_s + h2d_s + encode_s + sparsify_s + write_s
            rows.append(
                {
                    "docs": float(len(doc_ids)),
                    "wait_s": wait_s,
                    "h2d_s": h2d_s,
                    "encode_s": encode_s,
                    "sparsify_s": sparsify_s,
                    "write_s": write_s,
                    "total_step_s": total_step_s,
                    "peak_mem_gib": float(
                        torch.cuda.max_memory_allocated() / (1024**3)
                    ),
                }
            )

    if writer is not None:
        writer.finalize()

    local = {
        "rank": rank,
        "docs_total": sum(row["docs"] for row in rows),
        "step_time_total": sum(row["total_step_s"] for row in rows),
        "wait_mean": statistics.fmean(row["wait_s"] for row in rows),
        "h2d_mean": statistics.fmean(row["h2d_s"] for row in rows),
        "encode_mean": statistics.fmean(row["encode_s"] for row in rows),
        "sparsify_mean": statistics.fmean(row["sparsify_s"] for row in rows),
        "write_mean": statistics.fmean(row["write_s"] for row in rows),
        "peak_mem_gib": max(row["peak_mem_gib"] for row in rows),
    }

    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)

    if rank == 0:
        per_rank = [item for item in gathered if item is not None]
        total_docs = sum(float(item["docs_total"]) for item in per_rank)
        max_rank_time = max(float(item["step_time_total"]) for item in per_rank)
        result = {
            "world_size": world_size,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_mode": str(args.torch_compile_mode),
            "broadcast_buffers": bool(args.broadcast_buffers),
            "skip_write": bool(args.skip_write),
            "files_used": len(files),
            "aggregate_docs_per_sec": total_docs / max_rank_time,
            "per_rank_docs_per_sec": [
                float(item["docs_total"]) / float(item["step_time_total"])
                for item in per_rank
            ],
            "per_rank_peak_mem_gib": [float(item["peak_mem_gib"]) for item in per_rank],
            "mean_component_s": {
                "wait_s": statistics.fmean(float(item["wait_mean"]) for item in per_rank),
                "h2d_s": statistics.fmean(float(item["h2d_mean"]) for item in per_rank),
                "encode_s": statistics.fmean(float(item["encode_mean"]) for item in per_rank),
                "sparsify_s": statistics.fmean(float(item["sparsify_mean"]) for item in per_rank),
                "write_s": statistics.fmean(float(item["write_mean"]) for item in per_rank),
            },
        }
        print(json.dumps(result, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
