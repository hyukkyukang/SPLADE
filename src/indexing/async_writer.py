import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.multiprocessing as mp

from src.indexing.sparse_index import SparseShardWriter

logger: logging.Logger = logging.getLogger("src.indexing.async_writer")


@dataclass(frozen=True)
class SparseWriterConfig:
    output_dir: Path
    vocab_size: int
    rank: int
    top_k: int | None
    min_weight: float
    exclude_token_ids: list[int]
    shard_max_docs: int
    value_dtype: str


def _writer_worker_loop(
    cfg: SparseWriterConfig,
    queue_in: mp.Queue,
    error_queue: mp.Queue,
) -> None:
    try:
        writer = SparseShardWriter(
            output_dir=cfg.output_dir,
            vocab_size=cfg.vocab_size,
            rank=cfg.rank,
            top_k=cfg.top_k,
            min_weight=cfg.min_weight,
            exclude_token_ids=cfg.exclude_token_ids,
            shard_max_docs=cfg.shard_max_docs,
            value_dtype=cfg.value_dtype,
        )
        while True:
            item = queue_in.get()
            if item is None:
                break
            doc_ids, indptr, indices, values = item
            writer.write_sparse_csr_batch(doc_ids, indptr, indices, values)
        writer.finalize()
    except Exception as exc:  # pragma: no cover - defensive logging
        try:
            error_queue.put(exc)
        finally:
            logger.exception("Async writer process failed.")


class AsyncSparseWriter:
    """Process-based async writer to avoid GIL contention."""

    def __init__(
        self,
        cfg: SparseWriterConfig,
        *,
        queue_size: int,
        log: logging.Logger | None = None,
    ) -> None:
        self.cfg: SparseWriterConfig = cfg
        self.queue_size: int = max(1, int(queue_size))
        self.log: logging.Logger = log or logger
        self._ctx: mp.context.BaseContext = mp.get_context("spawn")
        self._queue: mp.Queue | None = None
        self._error_queue: mp.Queue | None = None
        self._process: mp.Process | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._queue = self._ctx.Queue(maxsize=self.queue_size)
        self._error_queue = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(
            target=_writer_worker_loop,
            args=(self.cfg, self._queue, self._error_queue),
            daemon=True,
        )
        self._process.start()
        self.log.info(
            "Started async writer process (pid=%s).", self._process.pid
        )

    def _raise_if_error(self) -> None:
        if self._error_queue is None:
            return
        try:
            exc = self._error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Async writer failed.") from exc

    def check_healthy(self) -> None:
        self._raise_if_error()
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError("Async writer process is no longer alive.")

    def write(
        self,
        doc_ids: Sequence[str],
        indptr: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._raise_if_error()
        if self._queue is None:
            raise RuntimeError("Async writer is not started.")
        payload = (list(doc_ids), indptr, indices, values)
        try:
            self._queue.put(payload, timeout=1.0)
        except queue.Full:
            self.log.warning(
                "Async writer queue is full; blocking until space is available."
            )
            self._queue.put(payload)

    def close(self) -> None:
        if self._queue is not None:
            self._queue.put(None)
        if self._process is not None:
            self._process.join()
            self.log.info(
                "Async writer process stopped (pid=%s).", self._process.pid
            )
        self._raise_if_error()
        if self._process is not None and self._process.exitcode not in (0, None):
            raise RuntimeError(
                f"Async writer exited with code {self._process.exitcode}."
            )
        self._queue = None
        self._error_queue = None
        self._process = None
