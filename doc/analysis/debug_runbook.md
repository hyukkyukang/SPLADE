# Debug Runbook

This runbook focuses on recurring failure modes in DDP, compile, and MLflow.

## DDP + Torch Compile Instability

Symptoms:
- segfault/illegal address under unfrozen DDP + max-autotune
- hangs after compile warmup

Checks:
- confirm strategy/static graph settings:
  - `training.strategy=ddp`
  - `training.static_graph=true`
  - `training.find_unused_parameters=false`
- confirm compile mode:
  - `training.torch_compile_mode=max-autotune`

Actions:
- use shared-encoder compile path (implemented in `compile_policy.py`)
- use fallback mode `max-autotune-no-cudagraphs` where needed
- if unstable, temporarily set:
  - `training.static_graph=false`
  - `training.torch_compile_mode=default`

## Training Progress Stalls

Symptoms:
- step number does not increase
- loss constant for long periods

Checks:

```bash
tail -n 50 log/train/<model>/<tag>/lightning_logs/version_0/metrics.csv
```

```bash
nvidia-smi
```

Actions:
- reduce per-rank batch size, then re-test (`bs=16, ga=2` style probes)
- verify dataloader workers/prefetch settings and dataset IO path
- ensure validation intervals are not too frequent for throughput

## MLflow Connectivity

Symptoms:
- run does not appear in server
- TLS/certificate/auth failures

Checks:
- `.env` contains:
  - `MLFLOW_TRACKING_URI`
  - optional TLS/cert env values
- config path:
  - `training.mlflow.enabled=true`
  - `training.mlflow.tracking_uri=<server>`

Actions:
- set `MLFLOW_TRACKING_URI=https://mlflow.hyukkyu.com`
- for local debug, disable MLflow:
  - `training.mlflow.enabled=false`
- verify network path and server availability before rerun
