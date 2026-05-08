from __future__ import annotations

import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.pl_module.common import build_model_tokenizer
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import get_logger
from src.utils.mlflow_utils import (
    add_mlflow_model_config_tags,
    build_mlflow_dataset_input_from_metadata,
    configure_mlflow_tls,
    has_logged_mlflow_dataset_inputs,
    has_logged_mlflow_model_outputs,
    log_mlflow_model_output,
    resolve_mlflow_logged_model_name,
    resolve_mlflow_model_type,
    resolve_mlflow_tags,
    sanitize_mlflow_metric_name,
)
from src.utils.model_utils import (
    apply_checkpoint_model_config,
    build_splade_model,
    load_splade_checkpoint,
)
from src.utils.mteb_search import MTEBSparseRetrievalAdapter
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    normalize_optional_path,
    normalize_optional_str,
    normalize_tag,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)


def _require_mteb() -> Any:
    try:
        import mteb
    except ImportError as exc:
        raise ImportError(
            "true MTEB evaluation requires the `mteb` package. Install it with `pip install mteb`."
        ) from exc
    return mteb


def _resolve_eval_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_source(cfg: DictConfig) -> tuple[str, str]:
    hf_model_path: str | None = normalize_optional_path(cfg.testing.hf_model_path)
    checkpoint_path: str | None = normalize_optional_path(cfg.testing.checkpoint_path)
    use_huggingface_model: bool = bool(cfg.mteb.use_huggingface_model)
    configured_hf_name: str | None = normalize_optional_str(cfg.model.huggingface_name)

    if hf_model_path is not None:
        if checkpoint_path is not None:
            raise ValueError(
                "Provide either testing.hf_model_path or testing.checkpoint_path, not both."
            )
        cfg.model.huggingface_name = hf_model_path
        return hf_model_path, "huggingface"

    if use_huggingface_model:
        if configured_hf_name is None:
            raise ValueError(
                "model.huggingface_name must be set when mteb.use_huggingface_model=true."
            )
        return configured_hf_name, "huggingface"

    if checkpoint_path is None:
        raise ValueError(
            "testing.checkpoint_path must be set unless Hugging Face model evaluation is enabled."
        )
    return checkpoint_path, "checkpoint"


def _resolve_tasks(cfg: DictConfig) -> tuple[list[Any], str]:
    mteb = _require_mteb()
    benchmark_name: str | None = normalize_optional_str(cfg.mteb.benchmark_name)
    task_names_cfg: Any = cfg.mteb.tasks
    task_names: list[str] = (
        [str(task_name) for task_name in task_names_cfg]
        if task_names_cfg is not None
        else []
    )
    if benchmark_name and task_names:
        raise ValueError("Provide either mteb.benchmark_name or mteb.tasks, not both.")

    if benchmark_name:
        benchmark: Any = mteb.get_benchmark(benchmark_name)
        tasks: list[Any] = list(benchmark.tasks)
        selection_name: str = benchmark_name
    else:
        if not task_names:
            raise ValueError(
                "Provide at least one task via mteb.tasks when mteb.benchmark_name is not set."
            )
        get_tasks_kwargs: dict[str, Any] = {"tasks": task_names}
        languages_cfg: Any = cfg.mteb.languages
        if languages_cfg is not None:
            get_tasks_kwargs["languages"] = [str(language) for language in languages_cfg]
        eval_splits_cfg: Any = cfg.mteb.eval_splits
        if eval_splits_cfg is not None:
            get_tasks_kwargs["eval_splits"] = [str(split) for split in eval_splits_cfg]
        tasks = list(mteb.get_tasks(**get_tasks_kwargs))
        selection_name = ",".join(task_names)

    non_retrieval_tasks: list[str] = [
        str(task.metadata.name)
        for task in tasks
        if str(task.metadata.type).lower() != "retrieval"
    ]
    if non_retrieval_tasks:
        raise ValueError(
            "true MTEB sparse evaluation currently supports retrieval tasks only. "
            f"Non-retrieval tasks requested: {', '.join(non_retrieval_tasks)}"
        )
    return tasks, selection_name


def _task_names(tasks: list[Any]) -> list[str]:
    return [str(task.metadata.name) for task in tasks]


def _resolve_visible_gpu_ids() -> list[str]:
    visible_devices_env: str | None = normalize_optional_str(
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visible_devices_env is not None:
        return [
            token.strip()
            for token in visible_devices_env.split(",")
            if token.strip()
        ]
    return [str(index) for index in range(int(torch.cuda.device_count()))]


def _resolve_parallel_gpu_ids(cfg: DictConfig) -> list[str]:
    configured_gpu_ids: Any = cfg.mteb.parallel.gpu_ids
    if configured_gpu_ids is not None:
        return [str(gpu_id) for gpu_id in configured_gpu_ids]
    return _resolve_visible_gpu_ids()


def _partition_task_names(task_names: list[str], worker_count: int) -> list[list[str]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    partitions: list[list[str]] = [[] for _ in range(worker_count)]
    task_index: int
    task_name: str
    for task_index, task_name in enumerate(task_names):
        partitions[task_index % worker_count].append(task_name)
    return [partition for partition in partitions if partition]


def _should_run_parallel(
    cfg: DictConfig,
    *,
    tasks: list[Any],
    device: torch.device,
    gpu_ids: list[str],
) -> bool:
    if bool(cfg.testing.use_cpu):
        return False
    if not bool(cfg.mteb.parallel.enabled):
        return False
    if device.type != "cuda":
        return False
    if len(tasks) <= 1:
        return False
    return len(gpu_ids) > 1


def _build_mteb_model(
    cfg: DictConfig,
    *,
    device: torch.device,
    model_source: str,
    model_source_kind: str,
) -> MTEBSparseRetrievalAdapter:
    use_cpu_for_model_build: bool = device.type == "cpu"
    model: SpladeModel = build_splade_model(
        cfg,
        use_cpu=use_cpu_for_model_build,
    )
    if model_source_kind == "checkpoint":
        checkpoint_path: str | None = normalize_optional_path(cfg.testing.checkpoint_path)
        if checkpoint_path is None:
            raise ValueError("testing.checkpoint_path must be set for checkpoint evaluation.")
        load_splade_checkpoint(model, checkpoint_path, logger=logger)
    model.to(device)
    model.eval()
    tokenizer = build_model_tokenizer(cfg.model)
    return MTEBSparseRetrievalAdapter(
        model=model,
        tokenizer=tokenizer,
        model_cfg=cfg.model,
        device=device,
        batch_size=int(cfg.mteb.batch_size),
        query_batch_size=(
            int(cfg.mteb.query_batch_size)
            if cfg.mteb.get("query_batch_size") is not None
            else None
        ),
        corpus_batch_size=(
            int(cfg.mteb.corpus_batch_size)
            if cfg.mteb.get("corpus_batch_size") is not None
            else None
        ),
        max_query_length=int(cfg.mteb.max_query_length),
        max_doc_length=int(cfg.mteb.max_doc_length),
        corpus_chunk_size=int(cfg.mteb.corpus_chunk_size),
        heartbeat_interval_seconds=float(cfg.mteb.heartbeat_interval_seconds),
        model_name=model_source,
        model_revision=normalize_optional_str(cfg.model.get("revision")),
    )


def _require_model_result_class() -> Any:
    mteb = _require_mteb()
    return mteb.results.ModelResult


def _evaluate_tasks(
    cfg: DictConfig,
    *,
    tasks: list[Any],
    selection_name: str,
    model_source: str,
    model_source_kind: str,
    device: torch.device,
) -> Any:
    log_message: str = (
        f"Running true MTEB retrieval evaluation on {model_source_kind}: {model_source}"
    )
    logger.info(log_message)
    logger.info("Selected MTEB benchmark/tasks: %s", selection_name)
    logger.info("Using evaluation device: %s", device)

    model: MTEBSparseRetrievalAdapter = _build_mteb_model(
        cfg,
        device=device,
        model_source=model_source,
        model_source_kind=model_source_kind,
    )

    encode_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.mteb.batch_size),
        "max_active_dims": (
            int(cfg.mteb.max_active_dims)
            if cfg.mteb.max_active_dims is not None
            else None
        ),
        "show_progress_bar": bool(cfg.mteb.show_progress_bar),
    }
    mteb = _require_mteb()
    return mteb.evaluate(
        model,
        tasks,
        encode_kwargs=encode_kwargs,
        cache=None,
        overwrite_strategy=(
            "always" if bool(cfg.mteb.overwrite_results) else "only-missing"
        ),
        show_progress_bar=bool(cfg.mteb.show_progress_bar),
        public_only=bool(cfg.mteb.public_only),
        num_proc=int(cfg.mteb.num_proc),
    )


def _log_results_summary(results: Any) -> None:
    task_result: Any
    for task_result in results.task_results:
        logger.info("%s main_score: %s", task_result.task_name, task_result.main_score)

    if results.task_results:
        mean_main_score: float = float(
            sum(float(task_result.main_score) for task_result in results.task_results)
            / len(results.task_results)
        )
        logger.info("mean_main_score: %s", mean_main_score)


def _is_mlflow_enabled(cfg: DictConfig) -> bool:
    if "mlflow" not in cfg:
        return False
    return bool(cfg.mlflow.get("enabled", True))


def _resolve_results_output_path(cfg: DictConfig) -> Path:
    return Path(cfg.log_dir) / "true_mteb_results.json"


def _save_results_if_requested(
    cfg: DictConfig,
    results: Any,
    *,
    force: bool = False,
) -> Path | None:
    if not force and not bool(cfg.mteb.save_json):
        return None
    output_path: Path = _resolve_results_output_path(cfg)
    results.to_disk(output_path)
    logger.info("Saved true MTEB results to %s", output_path)
    return output_path


def _build_mlflow_dataset_input(task_name: str) -> Any:
    resolved_name: str = str(task_name).strip() or "mteb_task"
    metadata: dict[str, str] = {
        "dataset_name": resolved_name,
        "benchmark": "MTEB",
        "task": "retrieval",
        "split": "test",
    }
    return build_mlflow_dataset_input_from_metadata(
        dataset_name=resolved_name,
        context="evaluation",
        metadata=metadata,
        source=f"https://huggingface.co/datasets/mteb/{resolved_name}",
    )


def _numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_numeric_metrics(results: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    task_results: list[Any] = list(getattr(results, "task_results", []) or [])
    main_scores: list[float] = []
    task_result_any: Any
    for task_result_any in task_results:
        numeric_main_score: float | None = _numeric_value(
            getattr(task_result_any, "main_score", None)
        )
        if numeric_main_score is not None:
            main_scores.append(numeric_main_score)
    if main_scores:
        mean_main_score: float = float(sum(main_scores) / len(main_scores))
        metrics["mean_main_score"] = mean_main_score

    task_result: Any
    for task_result in task_results:
        task_name: str = str(getattr(task_result, "task_name", "task"))
        main_score: float | None = _numeric_value(getattr(task_result, "main_score", None))
        if main_score is not None:
            metrics[f"{task_name}.main_score"] = main_score

        evaluation_time: float | None = _numeric_value(
            getattr(task_result, "evaluation_time", None)
        )
        if evaluation_time is not None:
            metrics[f"{task_name}.evaluation_time"] = evaluation_time

        scores_by_split: Any = getattr(task_result, "scores", None)
        if not isinstance(scores_by_split, dict):
            continue
        split_name: Any
        split_scores: Any
        for split_name, split_scores in scores_by_split.items():
            split_prefix: str = f"{task_name}.{str(split_name)}"
            if isinstance(split_scores, dict):
                split_scores = [split_scores]
            if not isinstance(split_scores, list):
                continue
            score_index: int
            score_payload: Any
            for score_index, score_payload in enumerate(split_scores):
                if not isinstance(score_payload, dict):
                    continue
                metric_prefix: str = split_prefix
                if len(split_scores) > 1:
                    metric_prefix = f"{split_prefix}.{score_index}"
                metric_name: Any
                metric_value: Any
                for metric_name, metric_value in score_payload.items():
                    numeric_metric: float | None = _numeric_value(metric_value)
                    if numeric_metric is None:
                        continue
                    metrics[f"{metric_prefix}.{str(metric_name)}"] = numeric_metric
    return metrics


def _build_mlflow_params(
    cfg: DictConfig,
    *,
    model_source: str,
    model_source_kind: str,
) -> dict[str, Any]:
    task_names_cfg: Any = cfg.mteb.tasks
    task_names: list[str] = (
        [str(task_name) for task_name in task_names_cfg]
        if task_names_cfg is not None
        else []
    )
    params: dict[str, Any] = {
        "benchmark": "true_mteb",
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "mteb.benchmark_name": normalize_optional_str(cfg.mteb.get("benchmark_name")),
        "mteb.tasks": ",".join(task_names),
        "mteb.batch_size": int(cfg.mteb.batch_size),
        "mteb.query_batch_size": (
            int(cfg.mteb.query_batch_size)
            if cfg.mteb.get("query_batch_size") is not None
            else None
        ),
        "mteb.corpus_batch_size": (
            int(cfg.mteb.corpus_batch_size)
            if cfg.mteb.get("corpus_batch_size") is not None
            else None
        ),
        "mteb.max_query_length": int(cfg.mteb.max_query_length),
        "mteb.max_doc_length": int(cfg.mteb.max_doc_length),
        "mteb.corpus_chunk_size": int(cfg.mteb.corpus_chunk_size),
        "mteb.heartbeat_interval_seconds": float(cfg.mteb.heartbeat_interval_seconds),
        "mteb.num_proc": int(cfg.mteb.num_proc),
        "mteb.public_only": bool(cfg.mteb.public_only),
        "mteb.use_huggingface_model": bool(cfg.mteb.use_huggingface_model),
        "mteb.parallel.enabled": bool(cfg.mteb.parallel.enabled),
    }
    max_active_dims: Any = cfg.mteb.get("max_active_dims")
    if max_active_dims is not None:
        params["mteb.max_active_dims"] = int(max_active_dims)
    max_workers: Any = cfg.mteb.parallel.get("max_workers")
    if max_workers is not None:
        params["mteb.parallel.max_workers"] = int(max_workers)
    gpu_ids: Any = cfg.mteb.parallel.get("gpu_ids")
    if gpu_ids is not None:
        params["mteb.parallel.gpu_ids"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    testing_hf_model_path: str | None = normalize_optional_path(cfg.testing.hf_model_path)
    if testing_hf_model_path is not None:
        params["testing.hf_model_path"] = testing_hf_model_path
    testing_checkpoint_path: str | None = normalize_optional_path(
        cfg.testing.checkpoint_path
    )
    if testing_checkpoint_path is not None:
        params["testing.checkpoint_path"] = testing_checkpoint_path
    return {
        key: value
        for key, value in params.items()
        if value is not None and value != ""
    }


def _log_mlflow_run_datasets_and_model(
    *,
    cfg: DictConfig,
    run_id: str,
    tracking_uri: str | None,
    task_names: list[str],
    model_source: str,
    model_source_kind: str,
) -> None:
    mlflow_client: MlflowClient = MlflowClient(tracking_uri=tracking_uri)
    run = mlflow_client.get_run(run_id)

    if not has_logged_mlflow_dataset_inputs(run):
        dataset_inputs: list[Any] = [
            _build_mlflow_dataset_input(task_name) for task_name in task_names
        ]
        if dataset_inputs:
            mlflow_client.log_inputs(run_id=run_id, datasets=dataset_inputs)

    if has_logged_mlflow_model_outputs(run):
        return

    model_cfg: DictConfig | None = cfg.get("model")
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("cfg.model must be set for MLflow model logging.")

    model_tags: dict[str, str] = {
        "benchmark": "true_mteb",
        "evaluator": "MTEB",
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "run_id": run_id,
    }
    hf_model_name: str | None = normalize_optional_str(model_cfg.get("huggingface_name"))
    if hf_model_name is not None:
        model_tags["huggingface_name"] = hf_model_name
    model_tags = add_mlflow_model_config_tags(model_tags, model_cfg)

    logged_model_name: str = resolve_mlflow_logged_model_name(model_cfg)
    log_mlflow_model_output(
        mlflow_client=mlflow_client,
        run=run,
        run_id=run_id,
        logged_model_name=logged_model_name,
        model_type=resolve_mlflow_model_type(model_cfg),
        model_tags=model_tags,
        tracking_uri=tracking_uri,
        step=0,
    )


def _log_to_mlflow(
    *,
    cfg: DictConfig,
    results: Any,
    task_names: list[str],
    model_source: str,
    model_source_kind: str,
    output_path: Path | None,
) -> None:
    if not _is_mlflow_enabled(cfg):
        return

    mlflow_cfg: DictConfig = cfg.mlflow
    configure_mlflow_tls(
        mlflow_cfg,
        sampling_interval_field="mlflow.system_metrics_sampling_interval",
        samples_before_logging_field="mlflow.system_metrics_samples_before_logging",
    )
    tracking_uri: str | None = normalize_optional_str(mlflow_cfg.get("tracking_uri"))
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name: str = (
        normalize_optional_str(mlflow_cfg.get("experiment_name")) or "Eval-True-MTEB"
    )
    run_name: str | None = normalize_optional_str(mlflow_cfg.get("run_name"))
    if run_name is None:
        run_name = normalize_tag(cfg.get("tag"))
    mlflow.set_experiment(experiment_name)

    tags: dict[str, str] = resolve_mlflow_tags(
        mlflow_cfg.get("tags"), field_name="mlflow.tags"
    )
    tags.setdefault("benchmark", "true_mteb")
    tags.setdefault("evaluator", "MTEB")
    tags.setdefault("model_source_kind", model_source_kind)
    tags.setdefault(
        "mteb.parallel.enabled",
        "true" if bool(cfg.mteb.parallel.enabled) else "false",
    )

    params: dict[str, Any] = _build_mlflow_params(
        cfg,
        model_source=model_source,
        model_source_kind=model_source_kind,
    )
    sanitized_metrics: dict[str, float] = {
        sanitize_mlflow_metric_name(f"true_mteb.{metric_name}"): float(metric_value)
        for metric_name, metric_value in _collect_numeric_metrics(results).items()
    }

    with mlflow.start_run(run_name=run_name) as active_run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        if sanitized_metrics:
            mlflow.log_metrics(sanitized_metrics)
        if bool(mlflow_cfg.get("log_artifacts", False)) and output_path is not None:
            if output_path.is_file():
                mlflow.log_artifact(output_path.as_posix())
            log_path: Path = Path(cfg.log_dir) / "evaluate_true_mteb.log"
            if log_path.is_file():
                mlflow.log_artifact(log_path.as_posix(), artifact_path="logs")

        run_id: str | None = (
            str(active_run.info.run_id)
            if active_run is not None and active_run.info is not None
            else None
        )
        if run_id is None:
            return
        _log_mlflow_run_datasets_and_model(
            cfg=cfg,
            run_id=run_id,
            tracking_uri=tracking_uri,
            task_names=task_names,
            model_source=model_source,
            model_source_kind=model_source_kind,
        )
        logger.info(
            "Logged true MTEB metrics to MLflow experiment=%s run_id=%s",
            experiment_name,
            run_id,
        )


def _prepare_worker_cfg(
    cfg: DictConfig,
    *,
    worker_tag: str,
    worker_task_names: list[str],
) -> DictConfig:
    worker_cfg: DictConfig = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True)
    )
    worker_cfg.tag = worker_tag
    worker_cfg.log_dir = str(Path(cfg.log_dir_base) / worker_tag)
    worker_cfg.mteb.benchmark_name = None
    worker_cfg.mteb.tasks = worker_task_names
    worker_cfg.mteb.save_json = True
    worker_cfg.mteb.parallel.enabled = False
    if "mlflow" in worker_cfg and worker_cfg.mlflow is not None:
        worker_cfg.mlflow.enabled = False
    return worker_cfg


def _merge_worker_results(
    worker_result_paths: list[Path],
    *,
    task_order: list[str],
) -> Any:
    ModelResult = _require_model_result_class()
    merged_results: list[Any] = []
    merged_exceptions: list[Any] = []
    model_name: str | None = None
    model_revision: str | None = None
    result_path: Path
    for result_path in worker_result_paths:
        worker_result: Any = ModelResult.from_disk(result_path)
        if model_name is None:
            model_name = str(worker_result.model_name)
        if model_revision is None:
            model_revision = worker_result.model_revision
        merged_results.extend(worker_result.task_results)
        if worker_result.exceptions:
            merged_exceptions.extend(worker_result.exceptions)

    order_map: dict[str, int] = {
        task_name: index for index, task_name in enumerate(task_order)
    }
    merged_results.sort(key=lambda task_result: order_map.get(task_result.task_name, 10**9))
    return ModelResult.model_construct(
        model_name=model_name or "no_model_name/available",
        model_revision=model_revision,
        task_results=merged_results,
        exceptions=merged_exceptions or None,
    )


def _run_parallel_tasks(
    cfg: DictConfig,
    *,
    tasks: list[Any],
    model_source: str,
    model_source_kind: str,
    device: torch.device,
    gpu_ids: list[str],
) -> Any:
    _ = model_source, model_source_kind, device
    base_tag: str = normalize_tag(cfg.get("tag")) or "true_mteb_parallel"
    max_workers_cfg: Any = cfg.mteb.parallel.max_workers
    max_workers: int = (
        int(max_workers_cfg)
        if max_workers_cfg is not None
        else len(gpu_ids)
    )
    worker_count: int = min(len(tasks), len(gpu_ids), max_workers)
    if worker_count <= 1:
        selection_name: str = ",".join(_task_names(tasks))
        return _evaluate_tasks(
            cfg,
            tasks=tasks,
            selection_name=selection_name,
            model_source=model_source,
            model_source_kind=model_source_kind,
            device=device,
        )

    task_names: list[str] = _task_names(tasks)
    partitions: list[list[str]] = _partition_task_names(task_names, worker_count)
    logger.info(
        "Running true MTEB task-parallel evaluation across %d GPUs: %s",
        len(partitions),
        ", ".join(gpu_ids[: len(partitions)]),
    )

    worker_processes: list[
        tuple[subprocess.Popen[str], Any, Path, list[str], str]
    ] = []
    worker_result_paths: list[Path] = []
    worker_index: int
    worker_task_names: list[str]
    for worker_index, worker_task_names in enumerate(partitions):
        worker_gpu_id: str = gpu_ids[worker_index]
        worker_tag: str = f"{base_tag}__worker{worker_index}"
        worker_log_dir: Path = Path(cfg.log_dir_base) / worker_tag
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        worker_cfg: DictConfig = _prepare_worker_cfg(
            cfg,
            worker_tag=worker_tag,
            worker_task_names=worker_task_names,
        )
        worker_cfg_path: Path = worker_log_dir / "worker_eval.yaml"
        OmegaConf.save(config=worker_cfg, f=str(worker_cfg_path))
        worker_console_path: Path = worker_log_dir / "worker_console.log"
        worker_console = worker_console_path.open("w", encoding="utf-8")
        command: list[str] = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config-path",
            str(worker_log_dir.resolve()),
            "--config-name",
            worker_cfg_path.stem,
        ]
        worker_env: dict[str, str] = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = worker_gpu_id
        process: subprocess.Popen[str] = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=worker_env,
            stdout=worker_console,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(
            "Worker %d assigned GPU %s for tasks: %s",
            worker_index,
            worker_gpu_id,
            ", ".join(worker_task_names),
        )
        worker_result_paths.append(worker_log_dir / "true_mteb_results.json")
        worker_processes.append(
            (
                process,
                worker_console,
                worker_console_path,
                worker_task_names,
                worker_gpu_id,
            )
        )

    process: subprocess.Popen[str]
    worker_console: Any
    worker_console_path: Path
    assigned_tasks: list[str]
    worker_gpu_id: str
    for process, worker_console, worker_console_path, assigned_tasks, worker_gpu_id in worker_processes:
        try:
            return_code: int = process.wait()
        finally:
            worker_console.close()
        if return_code == 0:
            continue
        log_tail: str = ""
        try:
            log_tail = worker_console_path.read_text(encoding="utf-8")[-4000:]
        except OSError:
            log_tail = ""
        raise RuntimeError(
            "true MTEB worker failed "
            f"(gpu={worker_gpu_id}, tasks={assigned_tasks}, exit_code={return_code}). "
            f"Worker log: {worker_console_path}\n{log_tail}"
        )

    merged_results: Any = _merge_worker_results(
        worker_result_paths,
        task_order=task_names,
    )
    return merged_results


def run_resolved(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    model_source: str
    model_source_kind: str
    model_source, model_source_kind = _resolve_model_source(cfg)
    checkpoint_exclude_keys: tuple[str, ...] = (
        "encode_path",
        "index_path",
        "encode_dir",
        "index_dir",
        "sparse_top_k",
        "sparse_min_weight",
        "max_input_length",
    )
    if model_source_kind == "huggingface":
        checkpoint_exclude_keys = ("huggingface_name",) + checkpoint_exclude_keys
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
        exclude_keys=checkpoint_exclude_keys,
    )

    tasks: list[Any]
    selection_name: str
    tasks, selection_name = _resolve_tasks(cfg)
    device: torch.device = _resolve_eval_device(bool(cfg.testing.use_cpu))
    gpu_ids: list[str] = _resolve_parallel_gpu_ids(cfg)
    results: Any
    if _should_run_parallel(cfg, tasks=tasks, device=device, gpu_ids=gpu_ids):
        results = _run_parallel_tasks(
            cfg,
            tasks=tasks,
            model_source=model_source,
            model_source_kind=model_source_kind,
            device=device,
            gpu_ids=gpu_ids,
        )
    else:
        results = _evaluate_tasks(
            cfg,
            tasks=tasks,
            selection_name=selection_name,
            model_source=model_source,
            model_source_kind=model_source_kind,
            device=device,
        )

    _log_results_summary(results)
    output_path: Path | None = _save_results_if_requested(
        cfg,
        results,
        force=_is_mlflow_enabled(cfg),
    )
    try:
        _log_to_mlflow(
            cfg=cfg,
            results=results,
            task_names=_task_names(tasks),
            model_source=model_source,
            model_source_kind=model_source_kind,
            output_path=output_path,
        )
    except Exception as exc:
        logger.warning("MLflow logging failed for true MTEB evaluation: %s", exc)


def run(cfg: DictConfig) -> None:
    run_resolved(cfg)


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate_true_mteb")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
