#!/usr/bin/env bash
# Psi0-style launch interface; checkpoint and model contract remain DiT4DiT.
set -euo pipefail
DIT4DIT_ROOT="${DIT4DIT_ROOT:-/share/2025slx/DIT4DIT}"
RTC_PYTHON="${RTC_PYTHON:-$DIT4DIT_ROOT/.venv_rtc_deploy/bin/python}"
: "${POLICY_GPU_UUID:?Select an available A100 UUID explicitly}"
export PYTHONPATH="$DIT4DIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if (( $# )); then
    exec "$RTC_PYTHON" -u "$(dirname "$0")/serve.py" "$@"
fi
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the run directory}"
: "${CHECKPOINT_STEP:?Set CHECKPOINT_STEP to an existing checkpoint step}"
exec "$RTC_PYTHON" -u "$(dirname "$0")/serve.py" \
    --host "${RTC_HOST:-127.0.0.1}" \
    --port "${RTC_PORT:-8014}" \
    --action_exec_horizon "${ACTION_EXEC_HORIZON:-30}" \
    --policy dit4dit \
    --rtc \
    --run-dir "$CHECKPOINT_DIR" \
    --ckpt-step "$CHECKPOINT_STEP" \
    --cosmos "${COSMOS_PATH:-/share/2025slx/models/Cosmos-Predict2.5-2B}"
