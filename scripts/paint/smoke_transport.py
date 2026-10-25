"""Finite transport integration smoke; run with Isaac Lab's Python, not MuJoCo."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ll-policy", type=Path, required=True)
parser.add_argument("--task", default="GO2-PIPER-PAINT-Transport-Flat")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=550)
parser.add_argument("--payload-mass", type=float, default=0.5)
parser.add_argument("--wrench-scale", type=float, default=0.1)
parser.add_argument("--mode", choices=("teacher", "student"), default="teacher")
parser.add_argument("--agent", choices=("zero", "reference"), default="reference")
parser.add_argument("--nominal", action="store_true")
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1 or not 0.001 <= args.payload_mass <= 10 or not 0 <= args.wrench_scale <= 1:
    parser.error("Expected positive envs/steps, payload mass in [.001,10], wrench scale in [0,1]")
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)

try:
    from datetime import datetime
    import json
    import gymnasium as gym
    import torch
    import LeggedManip_Lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.ll_policy_path = str(args.ll_policy.expanduser().resolve())
    cfg.observation_mode = args.mode
    cfg.wrench_scale = args.wrench_scale
    cfg.events.payload_mass.params["mass_range"] = (args.payload_mass, args.payload_mass)
    if args.nominal:
        for name in ("physics_material", "payload_material", "add_mass"):
            setattr(cfg.events, name, None)
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        cfg.events.reset_base.params["velocity_range"] = {key: (0, 0) for key in cfg.events.reset_base.params["velocity_range"]}
        cfg.events.reset_robot_joints.params["position_range"] = (1, 1)
        for actuator in cfg.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
        cfg.periodic_disturbance = False
    cfg.viewer.resolution = (640, 480)
    cfg.viewer.origin_type = "env"
    directory = Path("logs/paint_transport_smoke") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    env = wrapper.unwrapped
    try:
        obs, _ = wrapper.reset()
        action = env.action_manager.get_term("transport")
        parameters = [p.detach().clone() for p in action.ll_policy.parameters()]
        assert not any(p.requires_grad for p in action.ll_policy.parameters())
        assert env.action_manager.total_action_dim == 9
        expected = {"policy": 147 if args.mode == "teacher" else 144, "teacher": 147, "student": 144, "critic": 165}
        def check_observation(values):
            for key, width in expected.items():
                assert values[key].shape == (args.num_envs, width), (key, values[key].shape)
                assert torch.isfinite(values[key]).all(), key
        check_observation(obs)
        assert torch.count_nonzero(action.clean_history) == 0
        assert torch.count_nonzero(action.leg_actions) == 0
        if args.num_envs > 1:
            # Exercise reset of a subset after filling both LL and actor histories.
            command = torch.zeros(args.num_envs, 9, device=env.device)
            obs, _, _, _, _ = wrapper.step(command)
            saved_history, saved_target, saved_leg = (value[1:].clone() for value in
                                                      (action.clean_history, action.wrench_target, action.leg_actions))
            env._reset_idx(torch.tensor([0], device=env.device))
            assert torch.equal(action.clean_history[1:], saved_history)
            assert torch.equal(action.wrench_target[1:], saved_target)
            assert torch.equal(action.leg_actions[1:], saved_leg)
            assert torch.count_nonzero(action.clean_history[0]) == 0
            assert torch.count_nonzero(action.leg_actions[0]) == 0
        # Start recording only after reset-isolation checks, so their extra
        # resets cannot close the requested rollout video after a single frame.
        if args.video:
            wrapper = gym.wrappers.RecordVideo(wrapper, video_folder=str(directory / "video"),
                                               step_trigger=lambda step: step == 0,
                                               video_length=args.steps, disable_logger=True)
        # Restart all episodes for the measured rollout.
        obs, _ = wrapper.reset()
        terminated_count = timeout_count = 0
        max_grasp_error = 0.0
        payload_height_sum = 0.0
        min_base_height = float("inf")
        for step in range(args.steps):
            command = torch.zeros(args.num_envs, 9, device=env.device)
            if args.agent == "reference":
                command[:, :3] = action.reference()
            obs, reward, terminated, truncated, extras = wrapper.step(command)
            check_observation(obs)
            assert torch.isfinite(reward).all()
            assert extras["teacher_obs"].shape == (args.num_envs, 147)
            assert action.leg_actions.shape == (args.num_envs, 12)
            assert torch.isfinite(action.payload_forces).all()
            anchor = action.payload.data.root_pos_w + quat_apply(action.payload.data.root_quat_w, action.grasp_position)
            distance = torch.linalg.vector_norm(anchor - action.robot.data.body_pos_w[:, action.ee_id], dim=-1)
            max_grasp_error = max(max_grasp_error, distance.max().item())
            relative_payload = quat_apply_inverse(action.robot.data.root_quat_w,
                                                  action.payload.data.root_pos_w - action.robot.data.root_pos_w)
            payload_height_sum += relative_payload[:, 2].mean().item()
            base_height = action.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            min_base_height = min(min_base_height, base_height.min().item())
            assert distance.max() < 0.05, f"Grasp anchor error exceeds 5 cm: {distance.max().item()}"
            terminated_count += int(terminated.sum())
            timeout_count += int(truncated.sum())
            if step % 50 == 0:
                print(f"[TRANSPORT] step={step}; grasp_error_max={max_grasp_error:.4f}m; "
                      f"terminations={terminated_count}; timeouts={timeout_count}", flush=True)
        for saved, parameter in zip(parameters, action.ll_policy.parameters()):
            assert torch.equal(saved, parameter), "LL parameters changed"
        report = {"task": args.task, "steps": args.steps, "num_envs": args.num_envs, "mode": args.mode,
                  "observation_dims": expected, "hl_actions": 9, "ll_actions": 12, "ll_sha256": action.ll_sha256,
                  "payload_mass": args.payload_mass, "wrench_scale": args.wrench_scale, "nominal": args.nominal,
                  "max_grasp_error_m": max_grasp_error, "terminations": terminated_count, "timeouts": timeout_count,
                  "mean_payload_height_in_base_m": payload_height_sum / args.steps,
                  "min_base_height_above_env_origin_m": min_base_height}
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.steps >= 500 and timeout_count == 0:
            raise RuntimeError(f"No episode survived to timeout; physical stability needs investigation. Report: {directory}")
        if args.nominal and terminated_count:
            raise RuntimeError(f"Nominal smoke had {terminated_count} falls; inspect payload alignment/contact. Report: {directory}")
        print(f"PASS transport wiring and frozen LL checks: {directory / 'report.json'}", flush=True)
    finally:
        wrapper.close()
finally:
    launcher.app.close()
