"""Batch low-speed acceptance in Isaac Lab or MuJoCo; uses native-exported JIT.

    python scripts/paint/evaluate_low_speed.py --backend mujoco --policy POLICY --report OUT.json
    python scripts/paint/evaluate_low_speed.py --backend isaac --policy POLICY --report OUT.json --headless
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from low_speed_protocol import cases, command_at, initial_offsets, summarize, compare_baseline

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "isaac"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--payload-mass", type=float, default=0.)
    parser.add_argument("--quick", action="store_true", help="Skip transition tests; use --seeds 42 for a single-seed guard")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--require-targets", action="store_true")
    known, _ = parser.parse_known_args()
    if known.backend == "isaac":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 <= args.payload_mass <= 10 or len(set(args.seeds)) != len(args.seeds):
        parser.error("payload mass must be [0,10]; seeds must be unique")
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"Missing exported policy: {args.policy}")
    if args.report.exists():
        parser.error("Refusing to overwrite an existing report; choose a new path")
    return args


def load_policy(path, device):
    import torch
    with path.open("rb") as stream:
        policy = torch.jit.load(stream, map_location=device).eval()
    with torch.inference_mode():
        probe = policy(torch.zeros(2, 247, device=device))
    if probe.shape != (2, 12) or not torch.isfinite(probe).all():
        raise ValueError("Expected a finite feed-forward 247->12 exported policy")
    return policy


def mujoco_rollouts(args):
    import torch
    import yaml
    sys.path.insert(0, str(ROOT / "mujoco/deploy/deploy_mujoco/go2_piper"))
    from paint_loco import PaintBridge, mujoco
    from physics_audit import describe_model
    torch.set_num_threads(1)
    policy = load_policy(args.policy, "cpu")
    cfg = yaml.safe_load((ROOT / "mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.yaml").read_text())
    cfg["physics_profile"] = "point-contact"
    dt = cfg["simulation_dt"] * cfg["control_decimation"]
    steps = round(32. / dt)
    results = []
    execution = None
    for seed in args.seeds:
        for name, target in cases(args.quick):
            bridge = PaintBridge(cfg, args.payload_mass)
            offset, yaw = initial_offsets(seed)
            bridge.data.qpos[bridge.qpos_ids[:12]] += offset
            bridge.data.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
            mujoco.mj_forward(bridge.model, bridge.data)
            if execution is None:
                execution = describe_model(bridge)
            records = [[] for _ in range(6)]
            failed = 0
            for step in range(steps):
                t = step * dt
                command = command_at(name, target, t)
                with torch.inference_mode():
                    output = policy(torch.from_numpy(bridge.observation(command)).unsqueeze(0))[0].numpy()
                previous = bridge.previous_action.copy()
                bridge.set_action(output)
                for _ in range(cfg["control_decimation"]):
                    bridge.physics_step()
                    if not np.isfinite(bridge.data.qpos).all() or not np.isfinite(bridge.data.qvel).all():
                        failed = 1
                        break
                if failed:
                    break
                linear, angular, gravity = bridge.base_state()
                tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
                values = (t, command, [linear[0], linear[1], angular[2]], tilt,
                          bridge.previous_action - previous,
                          bridge.data.actuator_force[bridge.actuator_ids[:12]] * bridge.gear[:12])
                for buf, value in zip(records, values):
                    buf.append(np.array(value, copy=True))
                if tilt > 57.3 or bridge.data.xpos[bridge.base, 2] - bridge.floor_height < .12:
                    failed = 1
                    break
            result = summarize(name, target, *records[:6], failed, steps, dt)
            result["seed"] = seed
            results.append(result)
            print(f"[LOW-EVAL] {name} seed={seed} targets={result['passed']} rmse={result.get('steady_rmse')}", flush=True)
    return results, dt, execution


def isaac_rollouts(args):
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    try:
        import torch
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import LeggedManip_Lab.tasks  # noqa: F401
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
        from eval_options import configure_evaluation
        selected = [(seed, name, target) for seed in args.seeds for name, target in cases(args.quick)]
        task = "GO2-PIPER-PAINT-Loco-Flat-Play"
        cfg = parse_env_cfg(task, device=args.device, num_envs=len(selected))
        cfg.seed = 42
        cfg.episode_length_s = 40.
        cfg.commands.base_velocity.resampling_time_range = (100., 100.)
        configure_evaluation(cfg, SimpleNamespace(task=task, base_command=[0., 0., 0.],
                             payload_mass=args.payload_mass, no_disturbances=True,
                             no_domain_randomization=True, eval_name="low_speed_acceptance"))
        wrapper = gym.make(task, cfg=cfg)
        try:
            env = wrapper.unwrapped
            wrapper.reset()
            policy = load_policy(args.policy, env.device)
            robot = env.scene["robot"]
            from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.paint_locomotion import LEG_JOINT_NAMES
            leg_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
            q, dq = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
            pose = robot.data.root_state_w[:, :7].clone()
            for index, (seed, _, _) in enumerate(selected):
                offset, yaw = initial_offsets(seed)
                q[index, leg_ids] += torch.tensor(offset, device=env.device, dtype=q.dtype)
                pose[index, 3:] = pose.new_tensor([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
            robot.write_joint_state_to_sim(q, dq)
            robot.write_root_pose_to_sim(pose)
            env.sim.forward()
            env.scene.update(env.physics_dt)
            dt, steps = env.step_dt, round(32. / env.step_dt)
            logs = [[[] for _ in range(6)] for _ in selected]
            failed = np.zeros(len(selected), dtype=int)
            command_term = env.command_manager.get_term("base_velocity")
            previous = torch.zeros(len(selected), 12, device=env.device)
            for step in range(steps):
                if not launcher.app.is_running():
                    raise RuntimeError("Isaac app closed before acceptance completed")
                t = step * dt
                commands = np.asarray([command_at(name, target, t) for _, name, target in selected])
                command_term.vel_command_b[:] = torch.as_tensor(commands, device=env.device, dtype=torch.float32)
                with torch.inference_mode():
                    observation = env.observation_manager.compute()["policy"]
                    action = policy(observation).clamp(-4, 4)
                    _, _, terminated, timeout, _ = wrapper.step(action)
                done = (terminated | timeout).cpu().numpy()
                data = robot.data
                v = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1).cpu().numpy()
                tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)).rad2deg().cpu().numpy()
                delta = (action - previous).cpu().numpy()
                torque = data.applied_torque[:, leg_ids].cpu().numpy()
                for index in range(len(selected)):
                    if failed[index]:
                        continue
                    if done[index] or not np.isfinite(v[index]).all():
                        failed[index] = 1  # exclude automatic-reset state, retain failure
                        continue
                    for buf, value in zip(logs[index], (t, commands[index], v[index], tilt[index], delta[index], torque[index])):
                        buf.append(np.array(value, copy=True))
                previous = action.clone()
                if step % 250 == 0:
                    print(f"[LOW-EVAL] Isaac {t:.1f}/32s; parallel cases={len(selected)}", flush=True)
            results = []
            for index, (seed, name, target) in enumerate(selected):
                result = summarize(name, target, *logs[index], failed[index], steps, dt)
                result["seed"] = seed
                results.append(result)
            return results, dt, {"task": task, "nominal_physics": True, "delay_steps": 0,
                                 "seed_protocol": "42 nominal; other seeds matched leg/yaw perturbations, not training seeds"}
        finally:
            wrapper.close()
    finally:
        launcher.app.close()


def main():
    args = parse_args()
    results, dt, execution = mujoco_rollouts(args) if args.backend == "mujoco" else isaac_rollouts(args)
    report = dict(backend=args.backend, physics_profile="point-contact" if args.backend == "mujoco" else "nominal",
                  policy=str(args.policy), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  payload_mass_kg=args.payload_mass, seeds=args.seeds, quick=args.quick, dt=dt, duration_s=32.,
                  targets_passed=all(r["passed"] for r in results), results=results, execution=execution,
                  sampling="post-step; 32s full; constant-command steady 5..30s; transitions 8..11 and 21..24s")
    if args.baseline_report:
        report["preservation"] = compare_baseline(report, json.loads(args.baseline_report.read_text()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[LOW-EVAL] Report: {args.report}; targets_passed={report['targets_passed']}", flush=True)
    if args.fail_on_regression and not report.get("preservation", {}).get("passed", False):
        return 2
    if args.require_targets and not report["targets_passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
