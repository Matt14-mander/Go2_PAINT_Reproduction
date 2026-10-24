"""Server-side PAINT launcher. Use the Python from an existing Isaac Lab environment.

No SSH, environment creation or automatic dependency upgrades. Each failed stage
stops the sequence. All subprocesses use this interpreter and the repository root.
"""

import argparse
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
TASKS = {terrain: f"GO2-PIPER-PAINT-Loco-{terrain.title()}" for terrain in ("flat", "rough")}
EXPERIMENTS = {terrain: f"go2_piper_paint_loco{'_rough' if terrain == 'rough' else ''}" for terrain in TASKS}


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "flat", "rough", "pipeline"))
    parser.add_argument("--num-envs", type=positive_int, default=4096)
    parser.add_argument("--smoke-envs", type=positive_int, default=16)
    parser.add_argument("--smoke-steps", type=positive_int, default=100)
    parser.add_argument("--ppo-envs", type=positive_int, default=64)
    parser.add_argument("--ppo-iterations", type=positive_int, default=2)
    parser.add_argument("--flat-iterations", type=positive_int, default=5000)
    parser.add_argument("--rough-iterations", type=positive_int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="Exact PAINT LL checkpoint to resume; flat/rough stages only")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing or creating files")
    args = parser.parse_args(argv)
    if args.checkpoint and args.stage not in ("flat", "rough"):
        parser.error("--checkpoint is supported only for flat or rough")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must be cuda:<index>, for example cuda:0")
    return args


class Launcher:
    def __init__(self, args):
        self.args = args
        self.tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.log_dir = ROOT / "logs" / "paint_launcher" / self.tag

    def run(self, label, *arguments):
        command = [sys.executable, "-u", *map(str, arguments)]
        print(f"[{label}] {shlex.join(command)}", flush=True)
        if self.args.dry_run:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # tee output without buffering the entire Isaac log in memory.
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with (self.log_dir / f"{label}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if status:
            raise RuntimeError(f"{label} failed (exit {status}); see {self.log_dir / (label + '.log')}")

    def preflight(self):
        if self.args.dry_run:
            print("[preflight] Would check Python, Isaac packages, CUDA device, versions and LFS assets.")
            return
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(f"Repository requires Python 3.11; selected {sys.executable} ({sys.version.split()[0]})")
        for name in ("torch", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "isaacsim"):
            if importlib.util.find_spec(name) is None:
                raise RuntimeError(f"Missing {name}; activate the existing Isaac Lab environment first.")
        from packaging.version import Version
        import torch

        version = importlib.metadata.version("rsl-rl-lib")
        if Version(version) < Version("5.0.1"):
            raise RuntimeError(f"rsl-rl-lib {version} is incompatible; repository requires >=5.0.1")
        index = int(self.args.device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {self.args.device} is unavailable")
        asset_root = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper"
        assets = list(asset_root.rglob("*.usd"))
        if not assets:
            raise RuntimeError("GO2-PIPER USD assets missing; synchronize the complete source directory.")
        for path in assets:
            with path.open("rb") as stream:
                if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise RuntimeError(f"Unmaterialized LFS asset: {path}; run git lfs pull locally, then sync again.")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"python": sys.version, "executable": sys.executable, "rsl_rl": version,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": self.args.device, "gpu": torch.cuda.get_device_name(index),
                    "arguments": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(self.args).items()}}
        (self.log_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
        print(f"Launcher logs: {self.log_dir}", flush=True)

    def prepare(self):
        self.run("install_project", "-m", "pip", "install", "--no-deps", "-e", ROOT / "source/LeggedManip_Lab")

    def smoke(self):
        self.run("interface", "scripts/paint/check_interface.py")
        for terrain, task in TASKS.items():
            self.run(f"smoke_{terrain}", "scripts/paint/smoke_locomotion.py", "--task", task,
                     "--num_envs", self.args.smoke_envs, "--steps", self.args.smoke_steps,
                     "--device", self.args.device, "--headless")
            self.train(terrain, iterations=self.args.ppo_iterations, num_envs=self.args.ppo_envs, smoke=True)

    def train(self, terrain, *, iterations=None, num_envs=None, checkpoint=None, smoke=False):
        run_name = f"{'smoke' if smoke else 'train'}_{terrain}_{self.tag}"
        experiment = f"paint_smoke_{terrain}" if smoke else EXPERIMENTS[terrain]
        arguments = ["scripts/rsl_rl/train.py", "--task", TASKS[terrain],
                     "--num_envs", num_envs or self.args.num_envs,
                     "--max_iterations", iterations or getattr(self.args, f"{terrain}_iterations"),
                     "--device", self.args.device, "--seed", self.args.seed,
                     "--logger", "tensorboard", "--experiment_name", experiment,
                     "--run_name", run_name, "--headless", f"agent.device={self.args.device}"]
        if checkpoint is not None:
            checkpoint = Path(checkpoint).expanduser().resolve()
            # Use a local staging run instead of relying on upstream absolute
            # --load_run handling. Existing runs and checkpoints remain untouched.
            staging_name = f"resume_{terrain}_{self.tag}"
            staging = ROOT / "logs/rsl_rl" / experiment / staging_name
            print(f"[resume] Stage exact checkpoint {checkpoint} as {staging / 'checkpoint.pt'}", flush=True)
            if not self.args.dry_run:
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                staging.mkdir(parents=True, exist_ok=False)
                shutil.copy2(checkpoint, staging / "checkpoint.pt")
            arguments += ["--resume", "--load_run", staging_name, "--checkpoint", "checkpoint.pt"]
        self.run(f"{'ppo_smoke' if smoke else 'train'}_{terrain}", *arguments)
        return experiment, run_name

    def latest_checkpoint(self, experiment, run_name):
        root = ROOT / "logs/rsl_rl" / experiment
        runs = [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_" + run_name)]
        if len(runs) != 1:
            raise RuntimeError(f"Expected exactly one run for {run_name}; found {runs}")
        checkpoints = [(int(match.group(1)), path) for path in runs[0].glob("model_*.pt")
                       if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))]
        if not checkpoints:
            raise RuntimeError(f"No model checkpoint in {runs[0]}")
        return max(checkpoints, key=lambda item: item[0])[1]

    def execute(self):
        self.preflight()
        if self.args.stage == "prepare":
            self.prepare()
        elif self.args.stage == "smoke":
            self.smoke()
        elif self.args.stage in TASKS:
            self.train(self.args.stage, checkpoint=self.args.checkpoint)
        else:
            self.prepare()
            self.smoke()
            experiment, run_name = self.train("flat")
            checkpoint = (ROOT / "logs/DRY_RUN_FLAT_CHECKPOINT.pt" if self.args.dry_run
                          else self.latest_checkpoint(experiment, run_name))
            self.train("rough", checkpoint=checkpoint)
        print(f"Completed stage: {self.args.stage}", flush=True)


if __name__ == "__main__":
    try:
        Launcher(parse_args()).execute()
    except (RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
