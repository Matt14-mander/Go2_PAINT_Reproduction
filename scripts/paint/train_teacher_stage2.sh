#!/usr/bin/env bash
# Run inside the existing limx environment. No package installation or remote execution.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-smoke}"
LL="${LL:-$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-$PWD/logs/rsl_rl/go2_piper_paint_teacher_stage1/2026-10-26_12-12-23_stage1_fixed_arm/model_4999.pt}"
if [[ ! -d "$PWD/.runtime/rsl_rl_3_0_1/rsl_rl" ]]; then
  echo "Missing isolated .runtime/rsl_rl_3_0_1; use the environment that trained Stage1." >&2
  exit 1
fi
export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
[[ -f "$LL" ]] || { echo "Missing LL: $LL" >&2; exit 1; }
TASK="GO2-PIPER-PAINT-Transport-Flat-Stage2"
mkdir -p logs/teacher_stage2_console
STAMP="$(date +%Y%m%d_%H%M%S)"
case "$MODE" in
  smoke)
    python -u scripts/paint/smoke_transport.py \
      --task "$TASK" --ll-policy "$LL" --device cuda:0 \
      --num_envs 4 --steps 1600 --payload-mass 0.5 --wrench-scale 0.2 \
      --nominal --headless --kit_args="--/renderer/multiGpu/enabled=false" \
      2>&1 | tee "logs/teacher_stage2_console/smoke_$STAMP.log"
    ;;
  train)
    [[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage1 checkpoint: $STAGE1_CKPT" >&2; exit 1; }
    python -u scripts/rsl_rl/train.py \
      --task "$TASK" --warm-start "$STAGE1_CKPT" \
      --device cuda:0 --num_envs "${NUM_ENVS:-512}" --max_iterations "${ITERATIONS:-5000}" \
      --run_name stage2_small_arm --logger tensorboard --headless \
      --kit_args="--/renderer/multiGpu/enabled=false" \
      "env.ll_policy_path=$LL" \
      2>&1 | tee "logs/teacher_stage2_console/train_$STAMP.log"
    ;;
  *) echo "Usage: GPU=1 NUM_ENVS=512 bash $0 {smoke|train}" >&2; exit 2 ;;
esac
