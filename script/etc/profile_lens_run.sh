#!/usr/bin/env bash
# Profiling helper for an in-flight LENS training run.
#
# Why: phase1 GPU util oscillated 20-90% with rank processes pegged at 100%
# CPU. The hypothesis is per-step Python orchestration overhead (collation,
# multi-task sampling, DDP setup); this script lets you confirm by sampling
# the rank-0 Python stack with py-spy.
#
# kernel.yama.ptrace_scope=1 on this box, so py-spy needs CAP_SYS_PTRACE or
# sudo. There is no non-sudo path to a Python-level stack here (`/proc/PID/
# stack` is also root-only). Be ready for a sudo password prompt.
#
# Usage:
#   bash script/etc/profile_lens_run.sh              # default: dump
#   bash script/etc/profile_lens_run.sh dump         # snapshot rank-0 stacks
#   bash script/etc/profile_lens_run.sh dump --all   # snapshot every rank
#   bash script/etc/profile_lens_run.sh record       # 30s flame graph -> svg
#   bash script/etc/profile_lens_run.sh top          # live `top`-style view
#   bash script/etc/profile_lens_run.sh status       # /proc-only summary (no sudo)
#
# Env overrides:
#   PID=12345                pin a specific PID (skips auto-detect)
#   RECORD_SECS=60           seconds for the record mode
#   OUTPUT_DIR=/tmp          where flame graphs get written
#   PY_SPY=/path/to/py-spy   override py-spy binary

set -euo pipefail

PY_SPY="${PY_SPY:-$HOME/.local/bin/py-spy}"
RECORD_SECS="${RECORD_SECS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp}"

if [[ ! -x "${PY_SPY}" ]]; then
  echo "py-spy not found at ${PY_SPY}" >&2
  echo "install with:  pip install --user py-spy" >&2
  exit 1
fi

# --- find ranks ------------------------------------------------------------
discover_ranks() {
  # Match the worker procs (children of torch.distributed.run); the launcher
  # process itself uses ~0% CPU and is not interesting to sample.
  pgrep -af 'script/train_lens\.py' \
    | grep -v 'torch\.distributed\.run' \
    | awk '{print $1}'
}

if [[ -n "${PID:-}" ]]; then
  RANKS=("${PID}")
else
  mapfile -t RANKS < <(discover_ranks)
fi

if [[ ${#RANKS[@]} -eq 0 ]]; then
  echo "No train_lens.py worker processes found." >&2
  echo "Set PID=<pid> to target a specific process." >&2
  exit 1
fi

RANK0="${RANKS[0]}"

print_targets() {
  echo "Found ${#RANKS[@]} train_lens.py worker(s); rank-0 PID=${RANK0}"
}

# --- /proc-only status (no sudo) ------------------------------------------
proc_status() {
  print_targets
  echo
  printf '%-8s %-6s %-12s %-12s %-12s %s\n' \
    "PID" "STATE" "USER_CPU_s" "SYS_CPU_s" "VOL_CTXT" "INVOL_CTXT"
  for pid in "${RANKS[@]}"; do
    [[ -r "/proc/${pid}/stat" ]] || continue
    # /proc/PID/stat fields: utime(14), stime(15) in clock ticks.
    read -r -a fields < "/proc/${pid}/stat"
    local clk; clk="$(getconf CLK_TCK)"
    local utime_s sys_s
    utime_s=$(awk -v u="${fields[13]}" -v c="${clk}" 'BEGIN{printf "%.1f", u/c}')
    sys_s=$(awk -v s="${fields[14]}" -v c="${clk}" 'BEGIN{printf "%.1f", s/c}')
    local state; state="${fields[2]}"
    local vol invol
    vol=$(awk '/voluntary_ctxt_switches:/{print $2; exit}' "/proc/${pid}/status")
    invol=$(awk '/nonvoluntary_ctxt_switches:/{print $2; exit}' "/proc/${pid}/status")
    printf '%-8s %-6s %-12s %-12s %-12s %s\n' \
      "${pid}" "${state}" "${utime_s}" "${sys_s}" "${vol}" "${invol}"
  done
  echo
  echo "Hint: high involuntary_ctxt switches with low voluntary suggests"
  echo "      CPU-bound python (no I/O wait). High voluntary suggests"
  echo "      blocking on syscalls / locks / dataloader."
}

# --- py-spy modes (sudo required) -----------------------------------------
require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    echo "(sudo password required for ptrace; you'll be prompted)" >&2
  fi
}

cmd_dump() {
  require_sudo
  print_targets
  local targets=("${RANK0}")
  if [[ "${1:-}" == "--all" ]]; then
    targets=("${RANKS[@]}")
  fi
  for pid in "${targets[@]}"; do
    echo
    echo "=== py-spy dump pid=${pid} ==="
    sudo "${PY_SPY}" dump --pid "${pid}" || true
  done
}

cmd_record() {
  require_sudo
  print_targets
  mkdir -p "${OUTPUT_DIR}"
  local out="${OUTPUT_DIR}/lens_pyspy_$(date +%Y%m%d_%H%M%S)_pid${RANK0}.svg"
  echo "Recording rank-0 (pid=${RANK0}) for ${RECORD_SECS}s -> ${out}"
  sudo "${PY_SPY}" record \
    --pid "${RANK0}" \
    --duration "${RECORD_SECS}" \
    --rate 100 \
    --subprocesses \
    --idle \
    --output "${out}"
  chown "$(id -un):$(id -gn)" "${out}" 2>/dev/null || sudo chown "$(id -un):$(id -gn)" "${out}" || true
  echo "Done. Open the SVG in a browser; --idle includes off-CPU samples,"
  echo "so wide 'wait/sleep/recv' bars at the bottom = the rank is blocking."
}

cmd_top() {
  require_sudo
  print_targets
  echo "Press Ctrl-C to exit. Showing live function-level sampler for rank 0."
  sudo "${PY_SPY}" top --pid "${RANK0}"
}

cmd="${1:-dump}"
shift || true
case "${cmd}" in
  dump)   cmd_dump "$@" ;;
  record) cmd_record ;;
  top)    cmd_top ;;
  status) proc_status ;;
  *)
    echo "Unknown mode: ${cmd}" >&2
    echo "Modes: dump | dump --all | record | top | status" >&2
    exit 1
    ;;
esac
