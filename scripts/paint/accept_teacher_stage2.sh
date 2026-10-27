#!/usr/bin/env bash
# Windowed acceptance of the existing Stage2 checkpoint; never starts training.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-}"
RUN="${RUN:-$PWD/logs/rsl_rl/go2_piper_paint_teacher_stage2/2026-10-26_16-53-46_stage2_small_arm}"
LL="${LL:-$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$PWD/logs/teacher_stage2_acceptance/$STAMP}"
FAILED=0
case "$MODE" in mujoco|isaac) ;; *) echo "Usage: bash $0 {mujoco|isaac}" >&2; exit 2 ;; esac
mkdir -p "$OUT"
echo "Acceptance output: $OUT"

run_case() {
  local label="$1"
  shift
  if python -u "$@" 2>&1 | tee "$OUT/$label.log"; then
    echo "[ACCEPT] $label: process completed; inspect metrics before accepting."
  else
    echo "[ACCEPT] $label: FAILED; see $OUT/$label.log" >&2
    FAILED=1
  fi
}

if [[ "$MODE" = mujoco ]]; then
  # All forces act on the full payload; external-share divides by robot count.
  for CASE in side_pos side_neg yaw_pos yaw_neg repeat_forward; do
    PERIOD=()
    case "$CASE" in
      side_pos) FORCE=(0 16 0) ;;
      side_neg) FORCE=(0 -16 0) ;;
      yaw_pos) FORCE=(0 0 2) ;;
      yaw_neg) FORCE=(0 0 -2) ;;
      repeat_forward) FORCE=(16 0 0); PERIOD=(--force-period 15) ;;
    esac
    run_case "$CASE" mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
      --ll-policy "$LL" --controller teacher \
      --hl-policy "$RUN/exported/model_4999/policy.pt" --require-stage2-contract \
      --drive wrench --wrench-source external-share --audit-grasp \
      --payload-mass 0.5 --force-world "${FORCE[@]}" "${PERIOD[@]}" \
      --max-speed 0.35 --max-yaw-rate 0.4 --duration 60 --camera-azimuth 25 \
      --report "$OUT/$CASE.json"
  done
else
  if [[ ! -d "$PWD/.runtime/rsl_rl_3_0_1/rsl_rl" ]]; then
    echo "Missing isolated RSL-RL 3.0.1 environment" >&2
    exit 1
  fi
  export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
  unset CUDA_VISIBLE_DEVICES
  export HEADLESS=0 LIVESTREAM=0
  for SEED in 42 43 44; do
    for MASS in 0.25 0.5 1.0; do
      for CASE in force zero; do
        SCALE=0.2
        if [[ "$CASE" = zero ]]; then SCALE=0.0; fi
        LABEL="accept_${STAMP}_s${SEED}_m${MASS/./p}_$CASE"
        run_case "$LABEL" scripts/rsl_rl/play.py \
          --task GO2-PIPER-PAINT-Transport-Flat-Stage2 \
          --checkpoint "$RUN/model_4999.pt" --device "${DEVICE:-cuda:1}" \
          --num_envs 4 --seed "$SEED" --payload-mass "$MASS" \
          --eval-steps 2250 --eval-name "$LABEL" \
          --export-dir "$RUN/exported/model_4999" \
          --kit_args="--/renderer/multiGpu/enabled=false" \
          "env.ll_policy_path=$LL" "env.wrench_scale=$SCALE"
        if ! python - "$RUN/evaluations" "$OUT" "$LABEL" <<'PY'
from pathlib import Path
import shutil
import sys
source, output, label = sys.argv[1:]
reports = sorted(Path(source).glob(label + '_*.json'))
if not reports:
    raise SystemExit('No evaluation JSON for ' + label)
shutil.copyfile(reports[-1], Path(output) / (label + '.json'))
PY
        then
          FAILED=1
        fi
      done
    done
  done
  echo "Isaac JSON reports: $RUN/evaluations/accept_${STAMP}_*.json"
fi
echo "Acceptance output: $OUT"
exit "$FAILED"
