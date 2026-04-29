#!/usr/bin/env bash
# Watcher: poll for Phase 1 to finish, then launch Phase 2.
#
# Polls every 2 minutes. When the Phase 1 worker processes are gone:
#   - reads the final step from metrics.csv and last.ckpt timestamp
#   - if final step >= MIN_STEP_OK (default 4500 of 5000), launches Phase 2
#   - if Phase 1 ended early/badly, logs the situation and does NOT launch
#     (so we don't stack a doomed Phase 2 on top of a Phase 1 crash)
#
# All decisions logged to $WATCHER_LOG. Phase 2 stdout/stderr -> $PHASE2_LOG.
#
# Usage (run with nohup so it survives terminal close):
#   nohup bash script/etc/watch_then_launch_phase2.sh > /dev/null 2>&1 &
#   tail -f $WATCHER_LOG    # to monitor decisions

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LENS_LOGS_DIR="/mnt/ex-disk-1/hyukkyukang/SPLADE/lens/logs"
WATCHER_LOG="${LENS_LOGS_DIR}/watcher_phase2_$(date +%Y%m%d_%H%M).log"
PHASE2_LOG="${LENS_LOGS_DIR}/phase2_launch_$(date +%Y%m%d_%H%M).log"

PHASE1_TAG="${PHASE1_TAG:-phase1_d4000_LR1e5_20260429_1005}"
PHASE1_RUN_DIR="${LENS_LOGS_DIR}/lens_mistral_cluster4k/${PHASE1_TAG}"
PHASE1_METRICS="${PHASE1_RUN_DIR}/lightning_logs/version_0/metrics.csv"

POLL_SECS="${POLL_SECS:-120}"
MIN_STEP_OK="${MIN_STEP_OK:-4500}"
TARGET_STEPS="${TARGET_STEPS:-5000}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${WATCHER_LOG}"
}

is_phase1_running() {
  pgrep -f "${PHASE1_TAG}" >/dev/null 2>&1
}

last_step() {
  [[ -r "${PHASE1_METRICS}" ]] || { echo 0; return; }
  # CSV: epoch,lr-AdamW,step,train/contrastive_loss,train/distill_loss,train/loss,train/task_type
  awk -F, 'NR>1 && $3 ~ /^[0-9]+$/ {print $3}' "${PHASE1_METRICS}" \
    | sort -n | tail -1
}

mkdir -p "${LENS_LOGS_DIR}"
log "watcher started (PID $$). polling every ${POLL_SECS}s for tag=${PHASE1_TAG}"
log "watcher log: ${WATCHER_LOG}"
log "phase 2 log: ${PHASE2_LOG}"

# --- wait for phase 1 to finish -------------------------------------------
while is_phase1_running; do
  sleep "${POLL_SECS}"
done

log "phase 1 process tree is gone. inspecting outcome..."
final_step=$(last_step)
log "final step recorded in metrics.csv = ${final_step} (target=${TARGET_STEPS}, min_ok=${MIN_STEP_OK})"

if [[ -d "${PHASE1_RUN_DIR}/checkpoints" ]]; then
  ckpt_count=$(find "${PHASE1_RUN_DIR}/checkpoints" -name "*.ckpt" 2>/dev/null | wc -l)
  log "checkpoints written = ${ckpt_count}"
fi

# --- gate Phase 2 on Phase 1 success --------------------------------------
if (( final_step < MIN_STEP_OK )); then
  log "ABORT: phase 1 ended at step ${final_step}, below MIN_STEP_OK=${MIN_STEP_OK}."
  log "       not launching phase 2 — investigate phase 1 first."
  exit 0
fi

log "phase 1 reached step ${final_step}. proceeding to launch phase 2."

# --- launch Phase 2 (full run, defaults from launch_lens_phase2.sh) -------
LAUNCH_SCRIPT="${REPO_ROOT}/script/etc/launch_lens_phase2.sh"
if [[ ! -x "${LAUNCH_SCRIPT}" ]]; then
  log "ERROR: ${LAUNCH_SCRIPT} not found or not executable."
  exit 1
fi

log "exec: bash ${LAUNCH_SCRIPT} (output -> ${PHASE2_LOG})"
nohup bash "${LAUNCH_SCRIPT}" > "${PHASE2_LOG}" 2>&1 &
phase2_pid=$!
log "phase 2 launched, pid=${phase2_pid}"
log "watcher exiting."
