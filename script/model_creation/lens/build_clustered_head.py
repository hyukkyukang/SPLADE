import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.prototype.embeddinggemma_lsr.cli import (
    apply_config_overrides,
    parser_default_values,
    resolve_torch_dtype,
)
from src.utils.compact_head import (
    COMPACT_HEAD_FILENAME,
    build_clustered_compact_head_payload,
)

_DEFAULT_ASSIGNMENTS_FILENAME: str = "lens_cluster_assignments.npy"
_DEFAULT_SOURCE_TOKEN_IDS_FILENAME: str = "lens_cluster_source_token_ids.npy"
_DEFAULT_METADATA_FILENAME: str = "lens_cluster_metadata.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster a CausalLM output head into latent lexical dimensions and "
            "write a clustered SPLADE compact-head artifact."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/model_creation/lens/hf_backbone",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output model dir. Defaults to --model-dir for in-place update.",
    )
    parser.add_argument("--cluster-count", type=int, default=4000)
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "faiss", "sklearn"],
    )
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize output-head rows before clustering.",
    )
    parser.add_argument(
        "--exclude-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude tokenizer special-token rows from clustering.",
    )
    parser.add_argument(
        "--save-assignments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write token-to-cluster assignments as .npy files.",
    )
    parser.add_argument(
        "--assignments-filename",
        type=str,
        default=_DEFAULT_ASSIGNMENTS_FILENAME,
    )
    parser.add_argument(
        "--source-token-ids-filename",
        type=str,
        default=_DEFAULT_SOURCE_TOKEN_IDS_FILENAME,
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default=_DEFAULT_METADATA_FILENAME,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model load dtype; clustering always runs on float32 rows.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _default_values() -> dict[str, Any]:
    return parser_default_values(_build_parser())


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    return apply_config_overrides(args, defaults=_default_values())


def _normalize_backend(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _build_model_kwargs(
    *,
    trust_remote_code: bool,
    local_files_only: bool,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "dtype": dtype,
    }
    if bool(local_files_only):
        model_kwargs["local_files_only"] = True
    return model_kwargs


def _build_tokenizer_kwargs(
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": bool(trust_remote_code),
    }
    if bool(local_files_only):
        tokenizer_kwargs["local_files_only"] = True
    return tokenizer_kwargs


def _extract_output_head_weight(model: PreTrainedModel) -> torch.Tensor:
    output_head: Any = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise ValueError("Model output head has no weight parameter.")
    weight: torch.Tensor = output_head.weight.detach().to(dtype=torch.float32, device="cpu")
    if weight.ndim != 2:
        raise ValueError("Expected a rank-2 output-head weight tensor.")
    return weight.contiguous()


def _resolve_source_token_ids(
    *,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    exclude_special_tokens: bool,
) -> np.ndarray:
    include_mask: np.ndarray = np.ones(int(vocab_size), dtype=bool)
    if bool(exclude_special_tokens):
        special_ids: list[int] = [
            int(token_id)
            for token_id in tokenizer.all_special_ids
            if 0 <= int(token_id) < int(vocab_size)
        ]
        if special_ids:
            include_mask[np.asarray(sorted(set(special_ids)), dtype=np.int64)] = False
    source_token_ids: np.ndarray = np.nonzero(include_mask)[0].astype(np.int64, copy=False)
    if int(source_token_ids.size) == 0:
        raise ValueError("No source token rows remain after applying exclusions.")
    return source_token_ids


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms: np.ndarray = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms: np.ndarray = np.maximum(norms, 1e-12)
    return matrix / safe_norms


def _run_faiss_kmeans(
    matrix: np.ndarray,
    *,
    cluster_count: int,
    niter: int,
    nredo: int,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    row_count, hidden_size = matrix.shape
    if int(cluster_count) > int(row_count):
        raise ValueError(
            "cluster_count must be <= the number of source rows used for clustering."
        )
    kmeans = faiss.Kmeans(
        int(hidden_size),
        int(cluster_count),
        niter=int(niter),
        nredo=int(nredo),
        seed=int(seed),
        verbose=bool(verbose),
    )
    kmeans.train(np.ascontiguousarray(matrix, dtype=np.float32))
    _, assignments = kmeans.index.search(matrix, 1)
    centroids: np.ndarray = np.asarray(kmeans.centroids, dtype=np.float32).reshape(
        int(cluster_count),
        int(hidden_size),
    )
    return centroids, assignments.reshape(-1).astype(np.int64, copy=False)


def _run_sklearn_kmeans(
    matrix: np.ndarray,
    *,
    cluster_count: int,
    niter: int,
    nredo: int,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=int(cluster_count),
        max_iter=int(niter),
        n_init=max(int(nredo), 1),
        random_state=int(seed),
        verbose=int(verbose),
    )
    assignments: np.ndarray = estimator.fit_predict(matrix).astype(np.int64, copy=False)
    centroids: np.ndarray = np.asarray(
        estimator.cluster_centers_,
        dtype=np.float32,
    )
    return centroids, assignments


def _run_kmeans(
    matrix: np.ndarray,
    *,
    backend: str,
    cluster_count: int,
    niter: int,
    nredo: int,
    seed: int,
    verbose: bool,
) -> tuple[str, np.ndarray, np.ndarray]:
    resolved_backend: str = _normalize_backend(backend)
    errors: list[str] = []
    if resolved_backend in {"auto", "faiss"}:
        try:
            centroids, assignments = _run_faiss_kmeans(
                matrix,
                cluster_count=cluster_count,
                niter=niter,
                nredo=nredo,
                seed=seed,
                verbose=verbose,
            )
            return "faiss", centroids, assignments
        except Exception as exc:
            errors.append(f"faiss: {exc}")
            if resolved_backend == "faiss":
                raise RuntimeError("FAISS KMeans failed.") from exc
    if resolved_backend in {"auto", "sklearn"}:
        try:
            centroids, assignments = _run_sklearn_kmeans(
                matrix,
                cluster_count=cluster_count,
                niter=niter,
                nredo=nredo,
                seed=seed,
                verbose=verbose,
            )
            return "sklearn", centroids, assignments
        except Exception as exc:
            errors.append(f"sklearn: {exc}")
            if resolved_backend == "sklearn":
                raise RuntimeError("scikit-learn KMeans failed.") from exc
    raise RuntimeError(
        "No clustering backend succeeded. "
        + "; ".join(errors)
    )


def _copy_self_contained_model_dir(
    *,
    output_dir: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _save_metadata(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()
    args = _apply_config_overrides(args)

    model_dir: Path = Path(str(args.model_dir))
    output_dir: Path = (
        Path(str(args.output_dir))
        if args.output_dir is not None and str(args.output_dir).strip()
        else model_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype: torch.dtype = resolve_torch_dtype(str(args.dtype))
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        str(model_dir),
        **_build_tokenizer_kwargs(
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        ),
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        **_build_model_kwargs(
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
            dtype=dtype,
        ),
    )
    model.eval()

    output_weight: torch.Tensor = _extract_output_head_weight(model)
    vocab_size: int = int(output_weight.shape[0])
    hidden_size: int = int(output_weight.shape[1])
    source_token_ids: np.ndarray = _resolve_source_token_ids(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        exclude_special_tokens=bool(args.exclude_special_tokens),
    )
    if int(args.cluster_count) <= 0:
        raise ValueError("cluster_count must be positive.")
    if int(args.cluster_count) > int(source_token_ids.size):
        raise ValueError(
            "cluster_count must be <= the number of source rows used for clustering."
        )

    source_token_ids_tensor: torch.Tensor = torch.from_numpy(source_token_ids).to(
        dtype=torch.long
    )
    source_matrix: np.ndarray = np.ascontiguousarray(
        output_weight[source_token_ids_tensor].numpy(),
        dtype=np.float32,
    )
    if bool(args.normalize_rows):
        source_matrix = _normalize_rows(source_matrix)

    backend_used: str
    centroids: np.ndarray
    assignments: np.ndarray
    backend_used, centroids, assignments = _run_kmeans(
        source_matrix,
        backend=str(args.backend),
        cluster_count=int(args.cluster_count),
        niter=int(args.niter),
        nredo=int(args.nredo),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    cluster_sizes: np.ndarray = np.bincount(
        assignments,
        minlength=int(args.cluster_count),
    )

    payload: dict[str, Any] = build_clustered_compact_head_payload(
        weight=torch.from_numpy(centroids),
        extra_metadata={
            "source_vocab_size": int(vocab_size),
            "source_row_count": int(source_token_ids.size),
            "hidden_size": int(hidden_size),
            "cluster_count": int(args.cluster_count),
            "cluster_backend": backend_used,
            "normalize_rows": bool(args.normalize_rows),
        },
    )
    torch.save(payload, output_dir / COMPACT_HEAD_FILENAME)

    assignments_path: Path = output_dir / str(args.assignments_filename)
    source_token_ids_path: Path = output_dir / str(args.source_token_ids_filename)
    if bool(args.save_assignments):
        np.save(assignments_path, assignments)
        np.save(source_token_ids_path, source_token_ids)

    setattr(model.config, "splade_compact_head_file", COMPACT_HEAD_FILENAME)
    setattr(model.config, "splade_compact_head_alignment", "latent_cluster")
    setattr(model.config, "splade_compact_vocab_size", int(args.cluster_count))
    setattr(model.config, "lens_cluster_count", int(args.cluster_count))
    setattr(model.config, "lens_cluster_backend", backend_used)
    if bool(args.save_assignments):
        setattr(model.config, "lens_cluster_assignments_file", str(args.assignments_filename))
        setattr(
            model.config,
            "lens_cluster_source_token_ids_file",
            str(args.source_token_ids_filename),
        )

    if output_dir.resolve() != model_dir.resolve():
        _copy_self_contained_model_dir(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
        )
    else:
        model.config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    metadata: dict[str, Any] = {
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "cluster_backend": backend_used,
        "cluster_count": int(args.cluster_count),
        "niter": int(args.niter),
        "nredo": int(args.nredo),
        "seed": int(args.seed),
        "normalize_rows": bool(args.normalize_rows),
        "exclude_special_tokens": bool(args.exclude_special_tokens),
        "save_assignments": bool(args.save_assignments),
        "vocab_size": int(vocab_size),
        "source_row_count": int(source_token_ids.size),
        "excluded_row_count": int(vocab_size - source_token_ids.size),
        "hidden_size": int(hidden_size),
        "cluster_size_min": int(cluster_sizes.min().item()) if int(cluster_sizes.size) > 0 else 0,
        "cluster_size_max": int(cluster_sizes.max().item()) if int(cluster_sizes.size) > 0 else 0,
        "cluster_size_mean": float(cluster_sizes.mean().item()) if int(cluster_sizes.size) > 0 else 0.0,
        "compact_head_file": COMPACT_HEAD_FILENAME,
        "assignments_file": str(args.assignments_filename) if bool(args.save_assignments) else None,
        "source_token_ids_file": (
            str(args.source_token_ids_filename) if bool(args.save_assignments) else None
        ),
    }
    _save_metadata(output_dir / str(args.metadata_filename), metadata)

    print(f"Saved clustered compact head to {output_dir / COMPACT_HEAD_FILENAME}")
    print(
        f"Backend={backend_used}, clusters={int(args.cluster_count)}, "
        f"source_rows={int(source_token_ids.size)}, hidden_size={hidden_size}"
    )


if __name__ == "__main__":
    main()
