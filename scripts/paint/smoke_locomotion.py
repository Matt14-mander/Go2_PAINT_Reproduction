"""Finite headless integration check; run with the project's Isaac Lab Python."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="GO2-PIPER-PAINT-Loco-Flat")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import LeggedManip_Lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import SceneEntityCfg
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.paint_locomotion import (
    ARM_JOINT_NAMES, LEG_JOINT_NAMES, sample_payload_wrench,
)


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert env.action_manager.total_action_dim == 12
        assert obs["policy"].shape == (args.num_envs, 235), obs["policy"].shape
        robot = env.scene["robot"]
        _, names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        assert names == LEG_JOINT_NAMES, names
        arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        ee_ids, _ = robot.find_bodies("end_effector")
        assert len(ee_ids) == 1
        term = env.action_manager.get_term("joint_pos")
        ee_cfg = SceneEntityCfg("robot", body_names="end_effector")
        ee_cfg.resolve(env.scene)
        sample_payload_wrench(env, None, ee_cfg, zero_probability=0.0)
        # A partial action reset must not clear another robot's disturbance.
        if args.num_envs > 1:
            other_force = term.force_w[1:].clone()
            term.reset(torch.tensor([0], device=env.device))
            assert torch.equal(term.force_w[1:], other_force)
            assert torch.count_nonzero(term.force_w[0]) == 0
        obs, _ = env.reset()
        assert torch.count_nonzero(term.force_w) == 0
        sample_payload_wrench(env, None, ee_cfg, zero_probability=0.0)
        assert torch.allclose(robot.data.joint_pos[:, arm_ids], robot.data.default_joint_pos[:, arm_ids])
        masses = robot.root_physx_view.get_masses().cpu()
        # Physics backends may refresh default_mass after startup; check mass bounds
        # and print the actual masses instead of claiming a measured payload value.
        assert torch.isfinite(masses).all() and (masses > 0).all()
        print(f"End-effector total mass range: {masses[:, ee_ids[0]].min().item():.3f}.."
              f"{masses[:, ee_ids[0]].max().item():.3f} kg")
        resets = 0
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 12, device=env.device))
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            assert obs["policy"].shape == (args.num_envs, 235)
            assert torch.isfinite(env.scene["height_scanner"].data.ray_hits_w).all()
            assert torch.linalg.vector_norm(term.force_w, dim=-1).max() <= 40.001
            assert term.torque_w[..., 2].abs().max() <= 10.001
            assert torch.count_nonzero(term.force_w[..., 2]) == 0
            assert torch.allclose(
                robot.data.joint_pos_target[:, arm_ids], robot.data.default_joint_pos[:, arm_ids])
            resets += int((terminated | truncated).sum())
        print(f"PASS {args.task}: {args.steps} steps, policy=235, actions=12, episode resets={resets}")
        print("This checks wiring and finite physics, not the performance of an untrained zero-action policy.")
    finally:
        env.close()


try:
    main()
finally:
    launcher.app.close()
