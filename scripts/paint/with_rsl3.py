"""Run PAINT with a project-local official RSL-RL 3.0.1, leaving TRON2 intact.

Usage: with_rsl3.py install | check | run scripts/rsl_rl/play.py [arguments...]
Only install writes package files; run/check select that directory in a child
process's PYTHONPATH. No Conda packages, editable installs or shell startup files
are modified. Existing dependency versions are reused (--no-deps).
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / ".runtime/rsl_rl_3_0_1"
PROBE = """
import importlib.metadata as m
import importlib.util as u
import json
from pathlib import Path
import sys
target = Path(sys.argv[1]).resolve()
distribution = m.distribution('rsl-rl-lib')
spec = u.find_spec('rsl_rl')
origin = Path(spec.origin).resolve() if spec and spec.origin else None
metadata_root = Path(distribution.locate_file('')).resolve()
if distribution.version != '3.0.1' or metadata_root != target or origin is None or not origin.is_relative_to(target):
    raise RuntimeError(f'Wrong RSL selection: version={distribution.version}, metadata={metadata_root}, module={origin}')
print(json.dumps({'version': distribution.version, 'module': str(origin), 'metadata': str(metadata_root)}, indent=2))
"""


def selected_environment(target=TARGET):
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(target.resolve()) + (os.pathsep + previous if previous else "")
    return env


def verify(target=TARGET):
    if not (target / "rsl_rl/__init__.py").is_file():
        raise RuntimeError("Project-local RSL-RL is missing. Run: python scripts/paint/with_rsl3.py install")
    subprocess.run([sys.executable, "-c", PROBE, str(target)], cwd=ROOT,
                   env=selected_environment(target), check=True)


def install(target=TARGET):
    if target.exists() and any(target.iterdir()):
        # Never overwrite an unknown/partial existing installation.
        verify(target)
        print("Project-local RSL-RL 3.0.1 already installed.", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "--target", str(target), "rsl-rl-lib==3.0.1"], cwd=ROOT, check=True)
    verify(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check", "run"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action != "run" and args.arguments:
        parser.error("Only run accepts extra script arguments")
    if args.action == "install":
        install()
    elif args.action == "check":
        verify()
    else:
        if not args.arguments:
            parser.error("run requires a repository Python script")
        script = (ROOT / args.arguments[0]).resolve()
        if not script.is_relative_to(ROOT) or not script.is_file() or script.suffix != ".py":
            parser.error("run requires a Python script inside this repository")
        verify()
        os.chdir(ROOT)
        command = [sys.executable, str(script), *args.arguments[1:]]
        print(f"[PAINT] Interpreter: {sys.executable}; RSL directory: {TARGET}", flush=True)
        os.execve(sys.executable, command, selected_environment())


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
