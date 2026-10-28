"""Synchronous checkpoint guard. No training-environment steps or policy mutation.

Uses Isaac's native JIT exporter and a separate CPU-only MuJoCo process. A failed
preservation guard leaves checkpoints intact and stops training for inspection.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def install_regression_hook(runner, log_dir, baseline_policy, interval):
    from isaaclab_rl.rsl_rl import export_policy_as_jit
    root = Path(__file__).resolve().parents[2]
    output = Path(log_dir).resolve() / "regression"
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(baseline_policy).expanduser().resolve()
    baseline_copy = output / "original_policy.pt"
    shutil.copyfile(baseline, baseline_copy)
    evaluator = root / "scripts/paint/evaluate_low_speed.py"
    base_args = [sys.executable, "-u", str(evaluator), "--backend", "mujoco", "--quick", "--seeds", "42"]

    def evaluate(arguments, label):
        log = output / f"{label}.console.log"
        print(f"[LOW-GUARD] CPU MuJoCo regression: {label}; details: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(base_args + arguments, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Low-speed regression stopped training (exit={result.returncode}). Check {log}; saved checkpoints retained.")

    baseline_report = output / "baseline.json"
    evaluate(["--policy", str(baseline_copy), "--report", str(baseline_report)], "baseline")
    original_save = runner.save
    checked = set()
    last_checkpoint = [None]

    def check(checkpoint):
        path = Path(checkpoint)
        label = path.stem
        if label in checked:
            return
        destination = output / label
        policy = runner.alg.policy
        export_policy_as_jit(policy, normalizer=getattr(policy, "actor_obs_normalizer", None),
                             path=str(destination), filename="policy.pt")
        exported = destination / "policy.pt"
        manifest = dict(checkpoint=str(path.resolve()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        policy_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                        observations=247, actions=12, exporter="isaaclab_rl.export_policy_as_jit",
                        baseline_policy_sha256=hashlib.sha256(baseline_copy.read_bytes()).hexdigest())
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        evaluate(["--policy", str(exported), "--report", str(output / f"{label}.json"),
                  "--baseline-report", str(baseline_report), "--fail-on-regression"], label)
        checked.add(label)
        print(f"[LOW-GUARD] {label}: preservation passed; target acceptance is reported separately.", flush=True)

    def save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        last_checkpoint[0] = path
        iteration = int(Path(path).stem.removeprefix("model_"))
        if iteration == 0 or iteration % interval == 0:
            check(path)

    def finish():
        if last_checkpoint[0] is not None:
            check(last_checkpoint[0])

    runner.save = save
    runner.low_speed_regression_final = finish
