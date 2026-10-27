# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from runtime_checks import get_rsl_version_before_kit

# Kit may alter sys.path and distribution discovery. Resolve metadata with the
# user's active Python first, before importing/launching Isaac.
installed_version = get_rsl_version_before_kit()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from eval_options import add_eval_args, configure_evaluation

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--export-only", action="store_true", help="Load checkpoint, export actor, and exit without a rollout")
parser.add_argument("--export-dir", type=str, help="Explicit actor export directory, e.g. exported/model_9999")
parser.add_argument("--eval-steps", type=int, help="Stop after N control steps, also in a desktop window without recording")
parser.add_argument("--interface-snapshot", type=str, help="Save initialized live Loco 247->12 interface and actuator/body parameters")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
add_eval_args(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.eval_steps is not None and args_cli.eval_steps < 1:
    parser.error("--eval-steps must be positive")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

from packaging import version

"""Rest everything follows."""

import os
import time
import json
import hashlib
from datetime import datetime

from eval_metrics import LocoEvalMetrics, TransportEvalMetrics

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from LeggedManip_Lab.rsl_rl_compat import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import LeggedManip_Lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_evaluation(env_cfg, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # Older Isaac Lab releases have no isaaclab_rl.utils package. Local
        # checkpoints and run-directory lookup must not depend on this feature.
        try:
            from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        except ModuleNotFoundError as error:
            if error.name not in ("isaaclab_rl.utils", "isaaclab_rl.utils.pretrained_checkpoint"):
                raise
            raise RuntimeError(
                "This Isaac Lab installation does not provide published-checkpoint lookup. "
                "Omit --use_pretrained_checkpoint and pass --checkpoint /path/to/model.pt instead."
            ) from error
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    evaluation_name = f"{args_cli.eval_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": evaluation_name,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    print(f"[EVAL] Checkpoint loaded: {resume_path}", flush=True)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = (os.path.abspath(os.path.expanduser(args_cli.export_dir)) if args_cli.export_dir
                        else os.path.join(os.path.dirname(resume_path), "exported"))

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    def sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {"checkpoint": os.path.abspath(resume_path), "checkpoint_sha256": sha256_file(resume_path),
                "policy_sha256": sha256_file(os.path.join(export_model_dir, "policy.pt")),
                "task": args_cli.task, "rsl_rl_version": installed_version,
                "exported_at": datetime.now().isoformat(),
                "ll_policy_path": getattr(env_cfg, "ll_policy_path", None)}
    if "GO2-PIPER-PAINT-Transport-" in args_cli.task:
        manifest["transport_action"] = {
            "freeze_arm": env_cfg.actions.transport.freeze_arm,
            "command_limits": list(env_cfg.actions.transport.command_limits),
            **{key: getattr(env_cfg.actions.transport, key) for key in
               ("arm_scale", "arm_max_offset", "arm_max_speed", "settle_seconds", "force_ramp_seconds", "stop_hold_seconds")},
            "control_dt": env.unwrapped.step_dt,
            "command_deadband": env_cfg.actions.transport.command_deadband}
    with open(os.path.join(export_model_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"[EVAL] Policy export finished: {export_model_dir}", flush=True)
    if args_cli.interface_snapshot:
        if "GO2-PIPER-PAINT-Loco-" not in args_cli.task:
            raise ValueError("--interface-snapshot currently requires a PAINT Loco task")
        from interface_snapshot import save_interface_snapshot
        save_interface_snapshot(env, env.get_observations(), policy, args_cli.interface_snapshot, manifest)
    if args_cli.export_only:
        env.close()
        print("[EVAL] Export-only completed; no policy rollout performed.", flush=True)
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    metrics = LocoEvalMetrics(env.unwrapped) if "GO2-PIPER-PAINT-Loco-" in args_cli.task else None
    if "GO2-PIPER-PAINT-Transport-" in args_cli.task:
        metrics = TransportEvalMetrics(env.unwrapped)
    rollout_start = time.monotonic()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if metrics is not None:
                metrics.sample(actions)
            # print(obs["policy"])
            # env stepping
            obs, _, dones, extras = env.step(actions)
            if metrics is not None:
                metrics.record_resets(dones, extras)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if timestep == 1 or timestep % 100 == 0:
            print(f"[EVAL] step={timestep}; sim_time={timestep * dt:.1f}s; "
                  f"wall_time={time.monotonic() - rollout_start:.1f}s", flush=True)
        if args_cli.eval_steps is not None and timestep >= args_cli.eval_steps:
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if metrics is not None:
        report = metrics.summary()
        report.update(checkpoint=resume_path, steps=timestep, sim_time_s=timestep * dt,
                      wall_time_s=time.monotonic() - rollout_start,
                      fixed_command=args_cli.base_command, additional_payload_kg=args_cli.payload_mass,
                      no_disturbances=args_cli.no_disturbances,
                      nominal_physics=args_cli.no_domain_randomization)
        if isinstance(metrics, TransportEvalMetrics):
            report.update(wrench_scale=env_cfg.wrench_scale, periodic_disturbance=env_cfg.periodic_disturbance,
                          payload_mass_range=list(env_cfg.events.payload_mass.params["mass_range"]))
            report.pop("additional_payload_kg", None)
        directory = os.path.join(log_dir, "evaluations")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, evaluation_name + ".json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(f"[EVAL] Summary: {json.dumps(report)}\n[EVAL] Saved: {path}", flush=True)
    print(f"[EVAL] Finished after {timestep} steps; recording closed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
