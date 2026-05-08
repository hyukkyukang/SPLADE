import argparse
import os
import tempfile
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.profiler import ProfilerActivity, profile, record_function

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import EncodeDataModule
from src.model.pl_module import SPLADEEncodeModule
from src.utils.script_setup import configure_default_entrypoint_environment


configure_default_entrypoint_environment(load_env=True, set_matmul_precision=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the encode hot path with torch.profiler."
    )
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--data-file",
        type=str,
        default=".cache/hf/patent-us-corpus-small/data/later_legacy_full_v1-00000.parquet",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", type=str, default="default")
    parser.add_argument("--max-windows-per-forward", type=int, default=160)
    parser.add_argument("--row-limit", type=int, default=30)
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
        f"encoding.max_windows_per_forward={int(args.max_windows_per_forward)}",
        "encoding.async_write=false",
        "tag=profile_encode_torch_profiler",
        f"dataset.query_corpus_hf_data_files.train={str(args.data_file)}",
    ]
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


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    data_module = EncodeDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.predict_dataloader()
    iterator = iter(dataloader)
    module = setup_module(cfg)

    trace_dir = Path(tempfile.mkdtemp(prefix="torch-prof-encode-", dir="data"))
    trace_path = trace_dir / "trace.json"

    schedule = torch.profiler.schedule(
        wait=int(args.wait),
        warmup=int(args.warmup),
        active=int(args.active),
        repeat=int(args.repeat),
    )
    total_steps: int = (int(args.wait) + int(args.warmup) + int(args.active)) * int(
        args.repeat
    )

    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(total_steps):
            with record_function("splade.dataloader.next"):
                batch = next(iterator)
            with record_function("splade.batch.to_device"):
                doc_input_ids = batch["doc_input_ids"].to(
                    module.device, non_blocking=True
                )
                doc_attention_mask = batch["doc_attention_mask"].to(
                    module.device, non_blocking=True
                )
                doc_pooling_mask = batch.get("doc_pooling_mask")
                if doc_pooling_mask is not None:
                    doc_pooling_mask = doc_pooling_mask.to(
                        module.device, non_blocking=True
                    )
                doc_indptr = batch["doc_indptr"]
            with record_function("splade.encode_docs"):
                doc_reps = module._encode_and_aggregate_window_batch(
                    doc_input_ids,
                    doc_attention_mask,
                    doc_pooling_mask,
                    doc_indptr,
                )
            with record_function("splade.sparsify"):
                _ = module._sparsify_batch(doc_reps)
            torch.cuda.synchronize()
            prof.step()

    print(f"trace_path={trace_path}")
    print("TOP CPU")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=int(args.row_limit)
        )
    )
    print("TOP CUDA")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=int(args.row_limit)
        )
    )


if __name__ == "__main__":
    main()
