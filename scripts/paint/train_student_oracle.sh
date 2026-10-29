#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-smoke}"
GPU="${GPU:-1}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be numeric' >&2; exit 1; }
: "${TEACHER:?Set TEACHER to the accepted Stage2 native-exported policy.pt}"
LL="${LL:-$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt}"
unset CUDA_VISIBLE_DEVICES DISPLAY WAYLAND_DISPLAY
export CUDA_DEVICE_ORDER=PCI_BUS_ID LIVESTREAM=0 ENABLE_CAMERAS=0 PYTHONUNBUFFERED=1
RUN="$PWD/logs/paint_student_oracle/${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$(dirname "$RUN")"
APP=(--device "cuda:$GPU" --headless --kit_args="--no-window --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU")
COMMON=(--teacher "$TEACHER" --ll-policy "$LL")
case "$MODE" in
  smoke)
    python -u scripts/paint/student_oracle.py train "${COMMON[@]}" "${APP[@]}" \
      --output "$RUN" --num_envs 8 --iterations 3 --rollout-steps 256 \
      --teacher-warmup 1 --teacher-transition 1 --updates 2 --batch-size 128 --capacity 8192 --save-every 3 \
      2>&1 | tee "$RUN.console.log"
    python -u scripts/paint/student_oracle.py eval "${COMMON[@]}" "${APP[@]}" \
      --student "$RUN/exported/model_3/policy.pt" --output "${RUN}_eval" --num_envs 4 --steps 800 \
      2>&1 | tee "${RUN}_eval.console.log"
    echo "[STUDENT] SMOKE PASSED (pipeline only, not behavior acceptance): $RUN"
    ;;
  train)
    python -u scripts/paint/student_oracle.py train "${COMMON[@]}" "${APP[@]}" \
      --output "$RUN" --num_envs "${NUM_ENVS:-256}" --iterations "${ITERATIONS:-2000}" \
      2>&1 | tee "$RUN.console.log"
    ;;
  *) echo "Usage: TEACHER=... GPU=1 bash $0 {smoke|train}" >&2; exit 2 ;;
esac
