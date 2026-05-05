"""Aggregate a PyTorchProfiler Chrome-trace JSON into a bottleneck report.

Designed for the rank-0 trace produced by LENS training when
``LENS_PROFILE=1`` is set. Streams the (potentially multi-GB) JSON via
``ijson`` so we never load the full file into memory.

Two passes over the file (the first identifies the active-window time bounds
from ProfilerStep#N markers, the second aggregates events that fall inside
that window so warmup and tear-down don't skew totals).

Usage:
    python script/etc/analyze_lens_profile.py <path-to-rank0.pt.trace.json>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import ijson


def _scan_window(path: Path) -> tuple[int, int, int]:
    """Return (window_start_us, window_end_us, n_steps) using ProfilerStep markers."""
    step_starts: list[int] = []
    step_ends: list[int] = []
    with path.open("rb") as fh:
        for ev in ijson.items(fh, "traceEvents.item"):
            name = ev.get("name", "")
            if not name.startswith("ProfilerStep#"):
                continue
            ts = ev.get("ts")
            dur = ev.get("dur")
            if ts is None or dur is None:
                continue
            step_starts.append(int(ts))
            step_ends.append(int(ts) + int(dur))
    if not step_starts:
        return (0, 0, 0)
    return (min(step_starts), max(step_ends), len(step_starts))


def _aggregate(path: Path, window: tuple[int, int]) -> dict:
    """Aggregate complete events ('ph' == 'X') that fall inside the window."""
    win_lo, win_hi = window
    cat_total: dict[str, int] = defaultdict(int)         # inclusive dur per category
    op_total: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "dur": 0}
    )                                                     # (cat, name) -> {count, dur}
    nccl_total: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "dur": 0}
    )                                                     # NCCL/comm-only roll-up
    kernel_dur = 0
    memcpy_dur = 0
    cpu_op_dur = 0
    n_events = 0
    n_in_window = 0

    with path.open("rb") as fh:
        for ev in ijson.items(fh, "traceEvents.item"):
            n_events += 1
            if ev.get("ph") != "X":
                continue
            ts = ev.get("ts")
            dur = ev.get("dur")
            if ts is None or dur is None:
                continue
            ts = int(ts)
            dur = int(dur)
            if win_lo and (ts < win_lo or ts + dur > win_hi):
                continue
            n_in_window += 1
            cat = str(ev.get("cat", "?"))
            name = str(ev.get("name", "?"))
            cat_total[cat] += dur
            op_total[(cat, name)]["count"] += 1
            op_total[(cat, name)]["dur"] += dur

            if cat == "kernel":
                kernel_dur += dur
            elif cat == "gpu_memcpy":
                memcpy_dur += dur
            elif cat == "cpu_op":
                cpu_op_dur += dur

            # NCCL roll-up: catch nccl: kernels and c10d:: cpu_ops
            n_lc = name.lower()
            if (
                "nccl" in n_lc
                or n_lc.startswith("c10d::")
                or "all_gather" in n_lc
                or "all_reduce" in n_lc
                or "broadcast" in n_lc
                or "reduce_scatter" in n_lc
            ):
                nccl_total[name]["count"] += 1
                nccl_total[name]["dur"] += dur

    return {
        "n_events": n_events,
        "n_in_window": n_in_window,
        "cat_total": dict(cat_total),
        "op_total": dict(op_total),
        "nccl_total": dict(nccl_total),
        "kernel_dur": kernel_dur,
        "memcpy_dur": memcpy_dur,
        "cpu_op_dur": cpu_op_dur,
    }


def _fmt_us(us: int | float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us/1_000:.2f} ms"
    return f"{us:.0f} us"


def _print_top(label: str, items: list[tuple], top_n: int = 20) -> None:
    print(f"\n=== {label} (top {top_n}) ===")
    print(f"{'count':>8} {'total':>12} {'avg':>10}  name")
    print("-" * 96)
    for entry in items[:top_n]:
        if len(entry) == 3:
            count, total_us, name = entry
        else:
            (cat, name), stats = entry
            count = stats["count"]
            total_us = stats["dur"]
            name = f"[{cat}] {name}"
        avg_us = total_us / max(1, count)
        print(f"{count:>8} {_fmt_us(total_us):>12} {_fmt_us(avg_us):>10}  {name[:70]}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    trace_path = Path(sys.argv[1]).resolve()
    if not trace_path.is_file():
        print(f"trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    print(f"trace: {trace_path}  ({trace_path.stat().st_size/1e9:.2f} GB)")

    print("\n[1/2] scanning ProfilerStep markers...")
    win_lo, win_hi, n_steps = _scan_window(trace_path)
    if n_steps == 0:
        print("  no ProfilerStep markers found — analyzing full trace")
        window = (0, 0)
        wall_us = 0
    else:
        wall_us = win_hi - win_lo
        print(f"  found {n_steps} steps; active window {_fmt_us(wall_us)} "
              f"(avg step {_fmt_us(wall_us/n_steps)})")
        window = (win_lo, win_hi)

    print("\n[2/2] aggregating events in window...")
    agg = _aggregate(trace_path, window)
    print(f"  events scanned : {agg['n_events']:,}")
    print(f"  events in win  : {agg['n_in_window']:,}")

    # Headline numbers
    print("\n=== summary (within active profile window) ===")
    print(f"  wall time          : {_fmt_us(wall_us)}")
    print(f"  GPU kernel time    : {_fmt_us(agg['kernel_dur'])}"
          f"  ({100*agg['kernel_dur']/max(1,wall_us):.1f}% of wall)")
    print(f"  GPU memcpy time    : {_fmt_us(agg['memcpy_dur'])}"
          f"  ({100*agg['memcpy_dur']/max(1,wall_us):.1f}% of wall)")
    gpu_active = agg['kernel_dur'] + agg['memcpy_dur']
    print(f"  GPU active total   : {_fmt_us(gpu_active)}"
          f"  ({100*gpu_active/max(1,wall_us):.1f}% of wall)")
    print(f"  GPU idle estimate  : {_fmt_us(max(0, wall_us - gpu_active))}"
          f"  ({100*max(0,wall_us-gpu_active)/max(1,wall_us):.1f}% of wall)")
    print(f"  CPU op time (incl) : {_fmt_us(agg['cpu_op_dur'])}"
          f"  (note: nested, double-counts parents)")

    print("\n=== category roll-up (inclusive durations) ===")
    print(f"{'total':>12}  category")
    print("-" * 60)
    for cat, total in sorted(agg['cat_total'].items(), key=lambda x: -x[1]):
        print(f"{_fmt_us(total):>12}  {cat}")

    # NCCL roll-up — most relevant to the bottleneck question
    nccl_items = sorted(
        agg['nccl_total'].items(), key=lambda x: -x[1]["dur"]
    )
    if nccl_items:
        nccl_total_dur = sum(s["dur"] for _, s in nccl_items)
        print(f"\n=== NCCL / comm operations  "
              f"(combined {_fmt_us(nccl_total_dur)} = "
              f"{100*nccl_total_dur/max(1,wall_us):.1f}% of wall) ===")
        print(f"{'count':>8} {'total':>12} {'avg':>10}  name")
        print("-" * 96)
        for name, stats in nccl_items[:20]:
            avg = stats["dur"] / max(1, stats["count"])
            print(f"{stats['count']:>8} {_fmt_us(stats['dur']):>12} "
                  f"{_fmt_us(avg):>10}  {name[:70]}")

    # Top GPU kernels — what's actually crunching numbers
    kernels = [
        (key, stats) for key, stats in agg['op_total'].items()
        if key[0] == "kernel"
    ]
    kernels.sort(key=lambda x: -x[1]["dur"])
    _print_top("GPU kernels", kernels, top_n=20)

    # Top CPU ops (these double-count, but the SHAPE of the ranking is still
    # informative — wide eager ops will stand out from narrow C++ glue).
    cpu_ops = [
        (key, stats) for key, stats in agg['op_total'].items()
        if key[0] == "cpu_op"
    ]
    cpu_ops.sort(key=lambda x: -x[1]["dur"])
    _print_top("CPU ops (inclusive — double-counts nested calls)", cpu_ops, top_n=20)

    # Annotations & user-level markers (Lightning's training_step etc.)
    annot = [
        (key, stats) for key, stats in agg['op_total'].items()
        if key[0] in ("user_annotation", "python_function")
    ]
    annot.sort(key=lambda x: -x[1]["dur"])
    if annot:
        _print_top("User annotations / framework markers", annot, top_n=15)


if __name__ == "__main__":
    main()
