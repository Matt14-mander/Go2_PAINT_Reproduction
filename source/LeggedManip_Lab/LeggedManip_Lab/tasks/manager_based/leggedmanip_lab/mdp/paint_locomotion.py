"""PAINT replacement locomotion: leg policy, arm PD and persistent interaction loads.

The payload is approximated by additional mass and spherical inertia on the existing
end-effector rigid body. No payload geometry or grasp constraint is introduced here.
"""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


LEG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


class PaintLegPositionAction(JointPositionAction):
    """Expose 12 actions and hold the arm at its default pose on every physics step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._arm_ids, names = self._asset.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        if len(self._arm_ids) != 6 or self.action_dim != 12:
            raise ValueError(f"Expected 12 leg and 6 arm joints; got {self.action_dim}, {names}")
        # World-frame wrench buffers belong to the action, so reset is per environment.
        self.force_w = torch.zeros(env.num_envs, self._asset.num_bodies, 3, device=env.device)
        self.torque_w = torch.zeros_like(self.force_w)

    def process_actions(self, actions):
        # JointActionCfg.clip bounds processed joint targets, not raw actions.
        # Clip here as well as in the RSL wrapper for future direct HL consumers.
        super().process_actions(actions.clamp(-4.0, 4.0))

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[:, self._arm_ids], joint_ids=self._arm_ids
        )
        # Explicitly reapply every physics substep, including all-zero loads. Passing
        # every body also avoids body-index ambiguity in older wrench APIs.
        self._asset.set_external_force_and_torque(
            forces=self.force_w, torques=self.torque_w, is_global=True
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.force_w[env_ids] = 0.0
        self.torque_w[env_ids] = 0.0


@configclass
class PaintLegPositionActionCfg(JointPositionActionCfg):
    class_type: type = PaintLegPositionAction


def sample_payload_wrench(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    max_force: float = 40.0,
    max_yaw_torque: float = 10.0,
    zero_probability: float = 0.2,
):
    """Sample horizontal world-frame loads; keep them until the next sample/reset.

    Force magnitude is bounded by max_force, rather than bounding each component
    independently (which would permit a sqrt(2) times larger diagonal force).
    """
    if max_force < 0 or max_yaw_torque < 0 or not 0 <= zero_probability <= 1:
        raise ValueError("Invalid PAINT wrench sampling limits")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    term = env.action_manager.get_term("joint_pos")
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("PAINT wrench must resolve exactly one end-effector body")
    body_id = asset_cfg.body_ids[0]
    count = len(env_ids)
    angle = torch.rand(count, device=env.device) * (2.0 * torch.pi)
    radius = torch.sqrt(torch.rand(count, device=env.device)) * max_force
    active = torch.rand(count, device=env.device) >= zero_probability
    term.force_w[env_ids] = 0.0
    term.torque_w[env_ids] = 0.0
    term.force_w[env_ids, body_id, 0] = radius * torch.cos(angle) * active
    term.force_w[env_ids, body_id, 1] = radius * torch.sin(angle) * active
    term.torque_w[env_ids, body_id, 2] = (
        (2.0 * torch.rand(count, device=env.device) - 1.0) * max_yaw_torque * active
    )


def randomize_lumped_payload(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (0.0, 10.0),
    radius: float = 0.08,
):
    """Startup-only equivalent payload at the EE CoM, with consistent added inertia.

    This is a fixed load approximation, not a complete cooperative payload model.
    Add to the existing mass/inertia so the original end-effector never has zero mass.
    """
    if mass_range[0] < 0 or mass_range[1] < mass_range[0] or radius <= 0:
        raise ValueError("Invalid PAINT payload parameters")
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("PAINT payload must resolve exactly one end-effector body")
    ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else torch.as_tensor(env_ids, device="cpu")
    body_id = asset_cfg.body_ids[0]
    mass = torch.empty(len(ids), device="cpu").uniform_(*mass_range)
    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    masses[ids, body_id] += mass
    for diagonal in (0, 4, 8):
        inertias[ids, body_id, diagonal] += 0.4 * mass * radius**2
    asset.root_physx_view.set_masses(masses, ids)
    asset.root_physx_view.set_inertias(inertias, ids)


def terrain_relative_base_height(env, target_height: float = 0.30, std: float = 0.1):
    """Height reward relative to scanned ground, rather than absolute world Z."""
    hits = env.scene["height_scanner"].data.ray_hits_w[..., 2]
    hits = torch.where(torch.isfinite(hits), hits, env.scene["robot"].data.root_pos_w[:, 2:3] - target_height)
    height = env.scene["robot"].data.root_pos_w[:, 2] - hits.mean(dim=1)
    return torch.exp(-torch.square((height - target_height) / std))


def privileged_payload_wrench(env):
    """Training-only world wrench summary. Never expose true wrench to the actor."""
    term = env.action_manager.get_term("joint_pos")
    return torch.cat((term.force_w.sum(dim=1), term.torque_w.sum(dim=1)), dim=-1)
