"""Finite Isaac integration smoke; absence of low-speed tracking is NOT a smoke failure."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import torch
    import gymnasium as gym
    import LeggedManip_Lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.low_speed_math import sequence_factor
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.paint_locomotion import ARM_JOINT_NAMES
    task = "GO2-PIPER-PAINT-Loco-Flat-LowSpeed"
    cfg = parse_env_cfg(task, device=args.device, num_envs=16)
    wrapper = gym.make(task, cfg=cfg)
    try:
        env = wrapper.unwrapped
        with args.policy.open("rb") as stream:
            policy = torch.jit.load(stream, map_location=env.device).eval()
        obs, _ = wrapper.reset()
        assert obs["policy"].shape == (16, 247)
        assert env.action_manager.total_action_dim == 12
        term = env.command_manager.get_term("base_velocity")
        # Check deterministic sequence semantics on the actual manager adapter.
        term.mode[:] = 4
        term.target[:] = term.target.new_tensor([0., 0., .1])
        for age in (0., 2.5, 5.5, 7.5, 10.5):
            term.age[:] = age
            term._update_command()
            expected = term.target * sequence_factor(term.age.new_full((16,), age))[:, None]
            torch.testing.assert_close(term.command, expected)
        saved = term.target[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        torch.testing.assert_close(term.target[1:], saved)
        obs, _ = wrapper.reset()
        arm_ids, _ = env.scene["robot"].find_joints(ARM_JOINT_NAMES, preserve_order=True)
        assert len(arm_ids) == 6
        for step in range(650):  # crosses a 12-second command-resampling boundary
            with torch.inference_mode():
                action = policy(obs["policy"])
                assert action.shape == (16, 12) and torch.isfinite(action).all()
                obs, reward, terminated, timeout, _ = wrapper.step(action)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            assert (term.command.abs() <= term.command.new_tensor([1., 1., 1.5]) + 1e-6).all()
            if step % 100 == 0:
                print(f"[LOW-SMOKE] step={step}/650; finite 247->12; command modes={term.mode.bincount(minlength=5).tolist()}", flush=True)
        print("[LOW-SMOKE] Isaac manager/reset/sequence/finite rollout checks passed", flush=True)
    finally:
        wrapper.close()
finally:
    launcher.app.close()
