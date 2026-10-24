"""Tensor interface for a frozen PAINT LL policy, shared by future HL consumers.

No Isaac Sim imports. The caller supplies body-frame proprioception and the same
187 terrain ray heights as the training RayCaster (not world Z coordinates).
"""

import torch


def process_base_command(command: torch.Tensor, deadband: float = 0.05) -> torch.Tensor:
    """Clip [vx, vy, wz] to the trained envelope, then apply a per-axis deadband."""
    if command.shape[-1] != 3 or not 0 <= deadband < 1:
        raise ValueError("Expected (..., 3) base commands and deadband in [0, 1)")
    limits = command.new_tensor([1.0, 1.0, 1.5])
    clipped = torch.maximum(torch.minimum(command, limits), -limits)
    return torch.where(clipped.abs() < deadband, torch.zeros_like(clipped), clipped)


class PaintLocoInterface:
    """Own LL previous actions separately from HL actions; reset only done envs.

    Call build_observation once and record_action once per 0.02 s control step.
    The exported actor already includes the training observation normalizer.
    """

    def __init__(self, num_envs: int, device="cpu"):
        self.previous_action = torch.zeros(num_envs, 12, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.previous_action[env_ids] = 0.0

    def build_observation(self, lin_vel_b, ang_vel_b, gravity_b, command,
                          joint_pos_rel, joint_vel, height_scan):
        values = (lin_vel_b * 2.0, ang_vel_b * 0.2, gravity_b,
                  process_base_command(command), joint_pos_rel, joint_vel * 0.05,
                  self.previous_action, height_scan.clamp(-1.0, 1.0))
        widths = (3, 3, 3, 3, 12, 12, 12, 187)
        for value, width in zip(values, widths):
            if value.shape != (self.previous_action.shape[0], width):
                raise ValueError(f"Expected observation shape (N, {width}), got {value.shape}")
        observation = torch.cat(values, dim=-1)
        if not torch.isfinite(observation).all():
            raise ValueError("Non-finite PAINT locomotion observation")
        return observation

    def record_action(self, action):
        if action.shape != self.previous_action.shape or not torch.isfinite(action).all():
            raise ValueError("Expected finite (N, 12) leg actions")
        clipped = action.clamp(-4.0, 4.0)
        self.previous_action.copy_(clipped)
        return clipped
