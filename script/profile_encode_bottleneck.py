import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import EncodeDataModule
from src.index.sparse import SparseShardWriter
from src.model.pl_module import SPLADEEncodeModule
from src.utils.script_setup import configure_default_entrypoint_environment


configure_default_entrypoint_environment(load_env=True, set_matmul_precision=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile encode pipeline bottlenecks on a small corpus slice."
    )
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--data-file",
        type=str,
        default=".cache/hf/patent-us-corpus-small/data/later_legacy_full_v1-00000.parquet",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Skip sparse shard writing to isolate compute/input costs.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable encode torch.compile for profiling.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--max-windows-per-forward",
        type=int,
        default=None,
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    overrides = [
        "model=splade_v3_naver",
        "model.local_files_only=true",
        "dataset=patent_us_corpus_small",
        "encoding.long_doc_strategy=truncate",
        f"encoding.batch_size={int(args.batch_size)}",
        f"encoding.num_workers={int(args.num_workers)}",
        f"encoding.prefetch_factor={int(args.prefetch_factor)}",
        "encoding.num_devices=1",
        "encoding.strategy=single",
        f"encoding.torch_compile={'true' if bool(args.torch_compile) else 'false'}",
        f"encoding.torch_compile_mode={str(args.torch_compile_mode)}",
        "encoding.async_write=false",
        "tag=profile_encode_bottleneck",
        f"dataset.query_corpus_hf_data_files.train={str(args.data_file)}",
    ]
    if args.max_windows_per_forward is not None:
        overrides.append(
            f"encoding.max_windows_per_forward={int(args.max_windows_per_forward)}"
        )
    with initialize_config_dir(version_base=None, config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="encode", overrides=overrides)
    os.makedirs(cfg.log_dir, exist_ok=True)
    return cfg


def setup_module(cfg) -> SPLADEEncodeModule:
    torch.cuda.set_device(0)
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


def build_writer(module: SPLADEEncodeModule, cfg, output_dir: Path) -> SparseShardWriter:
    exclude_output_ids = (
        module._exclude_output_ids_tensor.tolist()
        if module._exclude_output_ids_tensor is not None
        else []
    )
    return SparseShardWriter(
        output_dir=output_dir,
        vocab_size=int(module.model.encoder.vocab_size),
        rank=0,
        top_k=cfg.encoding.sparse_top_k,
        min_weight=float(cfg.encoding.sparse_min_weight),
        exclude_output_ids=exclude_output_ids,
        source_exclude_token_ids=module._resolve_exclude_token_ids(),
        model_family=str(module.model.family),
        output_space=module.model.encoder.output_space,
        shard_max_docs=int(cfg.encoding.shard_max_docs),
        value_dtype=str(cfg.encoding.value_dtype),
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    data_module = EncodeDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.predict_dataloader()
    module = setup_module(cfg)

    writer: SparseShardWriter | None = None
    if not bool(args.skip_write):
        writer = build_writer(
            module,
            cfg,
            Path(tempfile.mkdtemp(prefix="splade-encode-prof-")),
        )

    iterator = iter(dataloader)
    rows: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()

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
                    "windows": float(int(doc_input_ids.shape[0])),
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

    docs_per_step = statistics.fmean(row["docs"] for row in rows)
    total_step_mean = statistics.fmean(row["total_step_s"] for row in rows)
    docs_per_sec = docs_per_step / total_step_mean

    breakdown: dict[str, dict[str, float]] = {}
    for key in ["wait_s", "h2d_s", "encode_s", "sparsify_s", "write_s"]:
        values = [row[key] for row in rows]
        mean_value = statistics.fmean(values)
        breakdown[key] = {
            "mean_s": mean_value,
            "pct_of_step": 100.0 * mean_value / total_step_mean,
            "median_s": statistics.median(values),
            "max_s": max(values),
        }

    result = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "steps_profiled": len(rows),
        "docs_per_step_mean": docs_per_step,
        "docs_per_sec_mean": docs_per_sec,
        "skip_write": bool(args.skip_write),
        "total_step_s": summarize([row["total_step_s"] for row in rows]),
        "peak_mem_gib_max": max(row["peak_mem_gib"] for row in rows),
        "breakdown": breakdown,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
