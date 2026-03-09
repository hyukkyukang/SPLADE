import concurrent.futures as cf
from pathlib import Path
from typing import Any, Callable


def run_shard_map_jobs(
    *,
    payloads: list[dict[str, Any]],
    worker_count: int,
    run_shard_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    shard_results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=int(worker_count)) as executor:
        future_by_shard: dict[cf.Future[dict[str, Any]], int] = {}
        payload: dict[str, Any]
        for payload in payloads:
            future: cf.Future[dict[str, Any]] = executor.submit(run_shard_fn, payload)
            future_by_shard[future] = int(payload["shard_index"])
        completed: int = 0
        total: int = len(payloads)
        future: cf.Future[dict[str, Any]]
        for future in cf.as_completed(future_by_shard):
            shard_index: int = future_by_shard[future]
            result: dict[str, Any] = future.result()
            shard_results.append(result)
            completed += 1
            print(
                f"[map-reduce] Completed shard {shard_index} "
                f"({completed}/{total}) docs_with_tokens={result['docs_with_tokens']}"
            )
    return shard_results


def cleanup_tmp_dir_if_empty(temp_dir: Path) -> None:
    try:
        temp_dir.rmdir()
    except OSError:
        pass
