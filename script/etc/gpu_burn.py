#!/usr/bin/env python3

import argparse
import logging
import math
import signal
import sys
import time
from multiprocessing.synchronize import Event as MpEvent
from types import FrameType
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp

LOGGER: logging.Logger = logging.getLogger("GpuBurner")

DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

DEFAULT_ALIGNMENT: int = 128
DEFAULT_DEVICE_SELECTION: str = "all"
DEFAULT_LOG_INTERVAL_SECONDS: float = 5.0
DEFAULT_MEMORY_FRACTION: float = 0.6
DEFAULT_WARMUP_STEPS: int = 10
MIN_MATRIX_SIZE: int = 256


class GpuBurner:
    def __init__(
        self,
        device_index: int,
        matrix_size: int,
        dtype: torch.dtype,
        log_interval_seconds: float,
        warmup_steps: int,
        allow_tf32: bool,
        memory_fraction: float,
        alignment: int,
        min_matrix_size: int,
        stop_event: MpEvent | None,
    ) -> None:
        self._device_index: int = device_index
        self._device: torch.device = torch.device(f"cuda:{device_index}")
        self._matrix_size: int = matrix_size
        self._dtype: torch.dtype = dtype
        self._log_interval_seconds: float = log_interval_seconds
        self._warmup_steps: int = warmup_steps
        self._memory_fraction: float = memory_fraction
        self._alignment: int = alignment
        self._min_matrix_size: int = min_matrix_size
        self._stop_event: MpEvent | None = stop_event
        self._stop_requested: bool = False

        # Ensure the target device is active before any allocations.
        torch.cuda.set_device(self._device)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        # Handle Ctrl+C / termination gracefully.
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum: int, frame: FrameType | None) -> None:
        _unused_signum: int = signum
        _unused_frame: FrameType | None = frame
        self._stop_requested = True

    def _auto_matrix_size(self) -> int:
        free_bytes: int
        total_bytes: int
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        _unused_total_bytes: int = total_bytes

        # Target a safe fraction of free memory to avoid OOM.
        target_bytes: int = int(float(free_bytes) * self._memory_fraction)
        bytes_per_element: int = torch.empty(
            (), dtype=self._dtype, device=self._device
        ).element_size()
        elements_per_matrix: int = max(target_bytes // (3 * bytes_per_element), 1)
        size: int = int(math.sqrt(elements_per_matrix))

        # Align for better kernel efficiency and consistency.
        size = max(size - (size % self._alignment), self._alignment)
        size = max(size, self._min_matrix_size)
        return size

    def _allocate_tensors(
        self, size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Allocate matrices on GPU to keep the loop allocation-free.
        a: torch.Tensor = torch.randn(
            (size, size), device=self._device, dtype=self._dtype
        )
        b: torch.Tensor = torch.randn(
            (size, size), device=self._device, dtype=self._dtype
        )
        c: torch.Tensor = torch.empty(
            (size, size), device=self._device, dtype=self._dtype
        )
        return a, b, c

    def _should_stop(self) -> bool:
        if self._stop_requested:
            return True
        if self._stop_event is None:
            return False
        return bool(self._stop_event.is_set())

    def run(self) -> int:
        if not torch.cuda.is_available():
            LOGGER.error("CUDA is not available. Please run on a CUDA-capable system.")
            return 1

        size: int = (
            self._matrix_size if self._matrix_size > 0 else self._auto_matrix_size()
        )
        a: torch.Tensor
        b: torch.Tensor
        c: torch.Tensor
        a, b, c = self._allocate_tensors(size)

        device_name: str = torch.cuda.get_device_name(self._device)
        LOGGER.info(
            "Using device=%s (index=%d), dtype=%s, matrix_size=%d",
            device_name,
            self._device_index,
            self._dtype,
            size,
        )

        # Warm up the kernels to stabilize performance.
        warmup_step: int
        for warmup_step in range(self._warmup_steps):
            _unused_warmup_step: int = warmup_step
            torch.matmul(a, b, out=c)
            a, c = c, a

        torch.cuda.synchronize(self._device)
        start_time: float = time.monotonic()
        last_log_time: float = start_time
        iterations: int = 0

        while not self._should_stop():
            torch.matmul(a, b, out=c)
            a, c = c, a
            iterations += 1

            now: float = time.monotonic()
            if (now - last_log_time) >= self._log_interval_seconds:
                torch.cuda.synchronize(self._device)
                elapsed: float = now - start_time
                it_per_sec: float = (
                    float(iterations) / elapsed if elapsed > 0.0 else 0.0
                )
                LOGGER.info(
                    "Iterations=%d, avg_it_per_sec=%.2f", iterations, it_per_sec
                )
                last_log_time = now

        torch.cuda.synchronize(self._device)
        total_elapsed: float = time.monotonic() - start_time
        LOGGER.info("Stopped after %.2fs", total_elapsed)
        return 0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_devices(
    devices_arg: str, legacy_device: int | None, device_count: int
) -> List[int]:
    if legacy_device is not None:
        return [legacy_device]
    devices_arg_normalized: str = devices_arg.strip().lower()
    if devices_arg_normalized == "all":
        return list(range(device_count))
    devices: List[int] = []
    entries: List[str] = devices_arg.split(",")
    for entry in entries:
        entry_text: str = entry.strip()
        if not entry_text:
            continue
        try:
            device_index: int = int(entry_text)
        except ValueError:
            LOGGER.error("Invalid device entry: %s", entry_text)
            return []
        devices.append(device_index)
    return devices


def _worker(
    device_index: int,
    matrix_size: int,
    dtype: torch.dtype,
    log_interval_seconds: float,
    warmup_steps: int,
    allow_tf32: bool,
    memory_fraction: float,
    alignment: int,
    min_matrix_size: int,
    stop_event: MpEvent,
) -> None:
    _configure_logging()
    burner: GpuBurner = GpuBurner(
        device_index=device_index,
        matrix_size=matrix_size,
        dtype=dtype,
        log_interval_seconds=log_interval_seconds,
        warmup_steps=warmup_steps,
        allow_tf32=allow_tf32,
        memory_fraction=memory_fraction,
        alignment=alignment,
        min_matrix_size=min_matrix_size,
        stop_event=stop_event,
    )
    burner.run()


def _build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Drive high GPU utilization with repeated GEMM operations."
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=DEFAULT_DEVICE_SELECTION,
        help="Comma-separated CUDA devices or 'all'.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Single CUDA device index (legacy option).",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=0,
        help="Square matrix size; 0 auto-selects based on free memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Tensor dtype used for GEMM.",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=DEFAULT_LOG_INTERVAL_SECONDS,
        help="Seconds between progress logs.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Number of warmup GEMM steps.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 for float32 matmul on supported GPUs.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=DEFAULT_MEMORY_FRACTION,
        help="Fraction of free GPU memory used for auto size.",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=DEFAULT_ALIGNMENT,
        help="Matrix size alignment for kernel efficiency.",
    )
    parser.add_argument(
        "--min-matrix-size",
        type=int,
        default=MIN_MATRIX_SIZE,
        help="Minimum matrix size for auto sizing.",
    )
    return parser


def main() -> int:
    _configure_logging()

    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()

    matrix_size: int = int(args.matrix_size)
    dtype_name: str = str(args.dtype)
    log_interval_seconds: float = float(args.log_interval_seconds)
    warmup_steps: int = int(args.warmup_steps)
    allow_tf32: bool = bool(args.allow_tf32)
    memory_fraction: float = float(args.memory_fraction)
    alignment: int = int(args.alignment)
    min_matrix_size: int = int(args.min_matrix_size)

    if not torch.cuda.is_available():
        LOGGER.error("CUDA is not available. Please run on a CUDA-capable system.")
        return 1

    device_count: int = torch.cuda.device_count()
    devices_arg: str = str(args.devices)
    legacy_device: int | None = args.device if args.device is not None else None
    devices: List[int] = _parse_devices(devices_arg, legacy_device, device_count)
    if not devices:
        LOGGER.error("No CUDA devices selected.")
        return 1

    for device_index in devices:
        device_id: int = int(device_index)
        if device_id < 0 or device_id >= device_count:
            LOGGER.error(
                "Invalid device index: %d (available: 0..%d)",
                device_id,
                device_count - 1,
            )
            return 1

    dtype: torch.dtype = DTYPE_MAP[dtype_name]
    # Use spawn for CUDA-safe multiprocessing.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Shared stop signal for all GPU workers.
    stop_event: MpEvent = mp.Event()
    processes: List[mp.Process] = []
    for device_index in devices:
        device_id: int = int(device_index)
        process: mp.Process = mp.Process(
            target=_worker,
            args=(
                device_id,
                matrix_size,
                dtype,
                log_interval_seconds,
                warmup_steps,
                allow_tf32,
                memory_fraction,
                alignment,
                min_matrix_size,
                stop_event,
            ),
            daemon=False,
        )
        process.start()
        processes.append(process)

    exit_code: int = 0
    try:
        for process in processes:
            process_handle: mp.Process = process
            process_handle.join()
    except KeyboardInterrupt:
        LOGGER.info("Stop requested. Shutting down workers.")
        stop_event.set()
        for process in processes:
            process_handle: mp.Process = process
            process_handle.join(timeout=5.0)
        for process in processes:
            process_handle: mp.Process = process
            if process_handle.is_alive():
                process_handle.terminate()
        for process in processes:
            process_handle: mp.Process = process
            process_handle.join()

    for process in processes:
        process_handle: mp.Process = process
        if process_handle.exitcode not in (0, None):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
