#!/usr/bin/env bash
# Existing limx environment only. Source weights and exported demo policy are read-only inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-smoke}"
ORIGINAL_RUN="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07"
SOURCE_CKPT="${SOURCE_CKPT:-$ORIGINAL_RUN/model_9999.pt}"
BASELINE_POLICY="${BASELINE_POLICY:-$ORIGINAL_RUN/exported/policy.pt}"
[[ -d .runtime/rsl_rl_3_0_1/rsl_rl ]] || { echo 'Missing isolated .runtime/rsl_rl_3_0_1' >&2; exit 1; }
[[ -f "$SOURCE_CKPT" && -f "$BASELINE_POLICY" ]] || { echo 'Missing source checkpoint / original exported policy' >&2; exit 1; }
export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
# Keep CUDA and Vulkan enumeration intact; select the device explicitly.
GPU="${GPU:-1}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be a numeric device index' >&2; exit 1; }
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# These runs have no cameras/windows. Do not inherit ThinLinc's virtual X display.
unset DISPLAY WAYLAND_DISPLAY
export LIVESTREAM=0
export ENABLE_CAMERAS=0
DEVICE="cuda:$GPU"
KIT_ARGS="--no-window --/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU"
echo "[LOWSPEED] device=$DEVICE; CUDA mask unset; display unset; headless"
export PYTHONUNBUFFERED=1
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CONSOLE="$PWD/logs/loco_low_speed_console"
mkdir -p "$CONSOLE"
TASK=GO2-PIPER-PAINT-Loco-Flat-LowSpeed
EXPERIMENT=go2_piper_paint_loco_low_speed
case "$MODE" in
  smoke)
    # First verify real Isaac command/reset/observation integration with original policy.
    python -u scripts/paint/smoke_low_speed.py --policy "$BASELINE_POLICY" \
      --device "$DEVICE" --headless --kit_args="$KIT_ARGS" \
      2>&1 | tee "$CONSOLE/smoke_env_$STAMP.log"
    # Then exercise weights-only transfer + real PPO update + native export + CPU regression.
    python -u scripts/rsl_rl/train.py --task "$TASK" --loco-warm-start "$SOURCE_CKPT" \
      --loco-baseline-policy "$BASELINE_POLICY" --loco-eval-interval 1 \
      --num_envs 32 --max_iterations 2 --device "$DEVICE" --seed 42 --logger tensorboard \
      --experiment_name go2_piper_paint_loco_low_speed_smoke --run_name "smoke_$STAMP" \
      --headless --kit_args="$KIT_ARGS" agent.device="$DEVICE" \
      2>&1 | tee "$CONSOLE/smoke_ppo_$STAMP.log"
    echo '[LOWSPEED] SMOKE PASSED: env, transfer, PPO, native export and preservation check.'
    ;;
  train|restore)
    ITERATIONS="${ITERATIONS:-3000}"
    if [[ "$MODE" == restore ]]; then
      [[ "$SOURCE_CKPT" != "$ORIGINAL_RUN/model_9999.pt" ]] || { echo 'restore requires SOURCE_CKPT=<accepted low-speed checkpoint>' >&2; exit 1; }
      TASK="${TASK}-Restore"
      EXPERIMENT=go2_piper_paint_loco_low_speed_restore
    fi
    python -u scripts/rsl_rl/train.py --task "$TASK" --loco-warm-start "$SOURCE_CKPT" \
      --loco-baseline-policy "$BASELINE_POLICY" --loco-eval-interval "${EVAL_INTERVAL:-500}" \
      --num_envs "${NUM_ENVS:-1024}" --max_iterations "$ITERATIONS" \
      --device "$DEVICE" --seed "${SEED:-42}" --logger tensorboard \
      --experiment_name "$EXPERIMENT" --run_name "${MODE}_$STAMP" \
      --headless --kit_args="$KIT_ARGS" agent.device="$DEVICE" \
      2>&1 | tee "$CONSOLE/${MODE}_$STAMP.log"
    ;;
  *) echo "Usage: GPU=1 NUM_ENVS=1024 bash $0 {smoke|train|restore}" >&2; exit 2 ;;
esac
