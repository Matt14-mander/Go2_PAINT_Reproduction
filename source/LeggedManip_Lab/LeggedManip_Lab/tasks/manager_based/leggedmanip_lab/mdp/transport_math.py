"""Tensor-only PAINT transport equations; importable without Isaac Sim."""

import torch


def wrench_profile(time, episode_length=10.0):
    """10% ramp-up, 80% hold, 10% ramp-down, clamped outside the episode."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    phase = time / episode_length
    return torch.minimum(phase / 0.1, (1.0 - phase) / 0.1).clamp(0.0, 1.0)


def process_transport_action(action, arm_default, arm_limits, deadband=0.05,
                             command_limits=(1.0, 1.0, 1.5), freeze_arm=False):
    if action.ndim != 2 or action.shape[1] != 9:
        raise ValueError("PAINT transport actions must be (N,9)")
    if not torch.isfinite(action).all():
        raise ValueError("Non-finite transport actions")
    limits = action.new_tensor(command_limits)
    if limits.shape != (3,) or not torch.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Expected three finite positive command limits")
    command = torch.maximum(torch.minimum(action[:, :3], limits), -limits)
    command = torch.where(command.abs() < deadband, 0.0, command)
    arm_action = action[:, 3:]
    arm_target = arm_default if freeze_arm else arm_default + 0.5 * arm_action
    arm_target = torch.maximum(torch.minimum(arm_target, arm_limits[..., 1]), arm_limits[..., 0])
    return command, arm_target, torch.cat((command, arm_action), dim=-1)


def locomotion_observation(linear, angular, gravity, command, joint_relative, joint_velocity, last_leg, heights):
    values = (linear * 2.0, angular * 0.2, gravity, command,
              joint_relative, joint_velocity * 0.05, last_leg, heights.clamp(-1.0, 1.0))
    widths = (3, 3, 3, 3, 18, 18, 12, 187)
    for value, width in zip(values, widths):
        if value.ndim != 2 or value.shape[-1] != width or value.shape[0] != linear.shape[0]:
            raise ValueError(f"Invalid LL observation component; expected (N,{width})")
    return torch.cat(values, dim=-1)


def tracking_terms(reference, base_linear, base_yaw_rate, literal_negative_exp=False):
    force = torch.exp(-(reference[:, :2] - base_linear[:, :2]).square().sum(-1) / 0.25)
    torque = torch.exp(-(reference[:, 2] - base_yaw_rate).square() / 0.25)
    # The tutorial lists negative weights next to exp rewards. Default to costs
    # so maximizing return encourages tracking; retain a literal reproduction switch.
    return (force, torque) if literal_negative_exp else (1.0 - force, 1.0 - torque)


def payload_height_cost(height):
    return ((0.05 - height).clamp_min(0) + (height - 0.25).clamp_min(0)).square()


def append_history(history, frame, mask):
    """Oldest to newest; update only selected environments, with no aliasing."""
    history[mask] = torch.cat((history[mask, 1:], frame[mask].unsqueeze(1)), dim=1)
